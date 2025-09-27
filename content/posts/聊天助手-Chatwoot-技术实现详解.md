---
title: "Chatwoot 技术实现详解"
date: 2024-12-20T10:00:00+08:00
draft: false
tags: ["Chatwoot", "Ruby", "Rails", "客服系统", "技术实现", "实时通信", "消息处理"]
categories: ["聊天助手"]
description: "深入分析Chatwoot项目的核心技术实现，包括服务层架构、消息处理流程、实时通信、自动化规则引擎、多渠道消息路由和通知系统等关键技术细节。"
keywords: ["Chatwoot", "客服系统", "Ruby on Rails", "WebSocket", "消息路由", "自动化规则", "通知系统"]
author: "技术分析师"
weight: 1
---



## 核心服务类详解

### 1. 自动化规则服务 (AutomationRules::ActionService)

```ruby
class AutomationRules::ActionService < ActionService
  # 初始化自动化规则执行服务
  # @param rule [AutomationRule] 自动化规则
  # @param account [Account] 账户
  # @param conversation [Conversation] 目标对话
  def initialize(rule, account, conversation)
    super(conversation)
    @rule = rule
    @account = account
    Current.executed_by = rule  # 设置执行上下文
  end

  # 执行自动化规则的所有动作
  def perform
    @rule.actions.each do |action|
      @conversation.reload  # 重新加载对话状态
      action = action.with_indifferent_access

      begin
        # 动态调用对应的动作方法
        send(action[:action_name], action[:action_params])
      rescue StandardError => e
        # 记录异常但不中断其他动作的执行
        ChatwootExceptionTracker.new(e, account: @account).capture_exception
      end
    end
  ensure
    Current.reset  # 清理执行上下文
  end

  private

  # 发送附件动作
  def send_attachment(blob_ids)
    return if conversation_a_tweet?  # Twitter不支持附件
    return unless @rule.files.attached?

    blobs = ActiveStorage::Blob.where(id: blob_ids)
    return if blobs.blank?

    params = { content: nil, private: false, attachments: blobs }
    Messages::MessageBuilder.new(nil, @conversation, params).perform
  end

  # 发送Webhook事件
  def send_webhook_event(webhook_url)
    payload = @conversation.webhook_data.merge(
      event: "automation_event.#{@rule.event_name}"
    )
    WebhookJob.perform_later(webhook_url[0], payload)
  end

  # 发送消息动作
  def send_message(message)
    return if conversation_a_tweet?

    params = {
      content: message[0],
      private: false,
      content_attributes: { automation_rule_id: @rule.id }
    }
    Messages::MessageBuilder.new(nil, @conversation, params).perform
  end

  # 添加私有备注
  def add_private_note(message)
    return if conversation_a_tweet?

    params = {
      content: message[0],
      private: true,
      content_attributes: { automation_rule_id: @rule.id }
    }
    Messages::MessageBuilder.new(nil, @conversation.reload, params).perform
  end

  # 发送团队邮件通知
  def send_email_to_team(params)
    teams = Team.where(id: params[0][:team_ids])

    teams.each do |team|
      TeamNotifications::AutomationNotificationMailer
        .conversation_creation(@conversation, team, params[0][:message])
        &.deliver_now
    end
  end
end
```

### 2. 渠道发送基础服务 (Base::SendOnChannelService)

这是所有渠道发送服务的基类，定义了统一的发送流程：

```ruby
class Base::SendOnChannelService
  # 初始化渠道发送服务
  # @param message [Message] 要发送的消息
  def initialize(message)
    @message = message
  end

  # 执行发送流程
  def perform
    validate_target_channel  # 验证目标渠道
    return unless outgoing_message?  # 只处理出站消息
    return if invalid_message?       # 跳过无效消息

    perform_reply  # 子类实现具体发送逻辑
  end

  private

  # 委托方法，简化访问路径
  delegate :conversation, to: :message
  delegate :contact, :contact_inbox, :inbox, to: :conversation
  delegate :channel, to: :inbox

  # 子类必须实现的抽象方法
  def channel_class
    raise 'Overwrite this method in child class'
  end

  def perform_reply
    raise 'Overwrite this method in child class'
  end

  # 检查是否为出站消息
  def outgoing_message?
    message.outgoing? || message.template?
  end

  # 检查消息是否无效
  def invalid_message?
    # 私有备注不发送到外部渠道
    # 避免消息循环(从渠道创建的出站消息)
    message.private? || outgoing_message_originated_from_channel?
  end

  # 检查出站消息是否来源于渠道本身
  def outgoing_message_originated_from_channel?
    # 来自外部渠道的消息会有source_id
    message.source_id.present?
  end

  # 验证目标渠道类型
  def validate_target_channel
    if inbox.channel.class != channel_class
      raise 'Invalid channel service was called'
    end
  end
end
```

### 3. WhatsApp消息处理服务

```ruby
class Whatsapp::IncomingMessageBaseService
  include ::Whatsapp::IncomingMessageServiceHelpers

  # 初始化WhatsApp入站消息处理服务
  # @param inbox [Inbox] 收件箱
  # @param params [Hash] Webhook参数
  def initialize(inbox, params)
    @inbox = inbox
    @params = params
  end

  # 处理WhatsApp Webhook事件
  def perform
    processed_params

    if processed_params.try(:[], :statuses).present?
      process_statuses  # 处理消息状态更新
    elsif processed_params.try(:[], :messages).present?
      process_messages  # 处理新消息
    end
  end

  private

  # 处理新消息
  def process_messages
    # 跳过不支持的消息类型(反应、临时消息等)
    return if unprocessable_message_type?(message_type)

    # 防止重复消息(Meta业务管理器配置错误可能导致重复Webhook)
    message_id = @processed_params[:messages].first[:id]
    return if find_message_by_source_id(message_id) || message_under_process?

    # 使用Redis缓存正在处理的消息ID，防止并发重复
    cache_message_source_id_in_redis
    set_contact
    return unless @contact

    ActiveRecord::Base.transaction do
      set_conversation      # 查找或创建对话
      create_messages      # 创建消息记录
      clear_message_source_id_from_redis  # 清理缓存
    end
  end

  # 处理消息状态更新(已发送、已送达、已读等)
  def process_statuses
    status = @processed_params[:statuses].first
    return unless find_message_by_source_id(status[:id])

    update_message_with_status(@message, status)
  rescue ArgumentError => e
    Rails.logger.error "Error while processing whatsapp status update #{e.message}"
  end

  # 更新消息状态
  def update_message_with_status(message, status)
    case status[:status]
    when 'sent'
      message.update!(status: :sent)
    when 'delivered'
      message.update!(status: :delivered)
    when 'read'
      message.update!(status: :read)
    when 'failed'
      message.update!(status: :failed)
      # 记录失败原因
      message.update!(
        content_attributes: message.content_attributes.merge(
          external_error: status[:errors]&.first
        )
      )
    end
  end
end
```

## 消息处理流程

### 消息构建器详细实现

```ruby
class Messages::MessageBuilder
  include ::FileTypeHelper
  attr_reader :message

  # 初始化消息构建器
  # @param user [User] 发送用户(客服)
  # @param conversation [Conversation] 所属对话
  # @param params [Hash] 消息参数
  def initialize(user, conversation, params)
    @params = params
    @private = params[:private] || false
    @conversation = conversation
    @user = user
    @message_type = params[:message_type] || 'outgoing'
    @attachments = params[:attachments]
    @automation_rule = content_attributes&.dig(:automation_rule_id)
    return unless params.instance_of?(ActionController::Parameters)

    @in_reply_to = content_attributes&.dig(:in_reply_to)
    @items = content_attributes&.dig(:items)
  end

  # 执行消息创建流程
  def perform
    @message = @conversation.messages.build(message_params)
    process_attachments
    process_emails
    @message.save!
    @message
  end

  private

  # 提取内容属性
  # - 转换ActionController::Parameters为普通哈希
  # - 尝试解析JSON字符串
  # - 返回空哈希如果内容不存在或解析错误
  def content_attributes
    params = convert_to_hash(@params)
    content_attributes = params.fetch(:content_attributes, {})

    return parse_json(content_attributes) if content_attributes.is_a?(String)
    return content_attributes if content_attributes.is_a?(Hash)

    {}
  end

  # 转换对象为哈希
  # 如果是ActionController::Parameters实例，转换为unsafe哈希
  # 否则返回原对象
  def convert_to_hash(obj)
    return obj.to_unsafe_h if obj.instance_of?(ActionController::Parameters)

    obj
  end

  # 尝试解析JSON字符串
  # 成功则返回符号化键名的哈希
  # 失败则返回空哈希
  def parse_json(content)
    JSON.parse(content, symbolize_names: true)
  rescue JSON::ParserError
    {}
  end

  # 处理附件上传
  def process_attachments
    return if @attachments.blank?

    @attachments.each do |uploaded_attachment|
      attachment = @message.attachments.build(
        account_id: @message.account_id,
        file: uploaded_attachment
      )

      attachment.file_type = if uploaded_attachment.is_a?(String)
                               file_type_by_signed_id(
                                 uploaded_attachment
                               )
                             else
                               file_type(uploaded_attachment&.content_type)
                             end
    end
  end

  # 处理邮件相关字段
  def process_emails
    return unless @conversation.inbox&.inbox_type == 'Email'

    cc_emails = process_email_string(@params[:cc_emails])
    bcc_emails = process_email_string(@params[:bcc_emails])
    to_emails = process_email_string(@params[:to_emails])

    all_email_addresses = cc_emails + bcc_emails + to_emails
    validate_email_addresses(all_email_addresses)

    @message.content_attributes[:cc_emails] = cc_emails
    @message.content_attributes[:bcc_emails] = bcc_emails
    @message.content_attributes[:to_emails] = to_emails
  end

  # 处理邮件地址字符串
  def process_email_string(email_string)
    return [] if email_string.blank?

    email_string.gsub(/\s+/, '').split(',')
  end

  # 验证邮件地址格式
  def validate_email_addresses(all_emails)
    all_emails&.each do |email|
      raise StandardError, 'Invalid email address' unless email.match?(URI::MailTo::EMAIL_REGEXP)
    end
  end

  # 确定消息类型
  def message_type
    if @conversation.inbox.channel_type != 'Channel::Api' && @message_type == 'incoming'
      raise StandardError, 'Incoming messages are only allowed in Api inboxes'
    end

    @message_type
  end

  # 确定发送者
  def sender
    message_type == 'outgoing' ? (message_sender || @user) : @conversation.contact
  end

  # 外部创建时间
  def external_created_at
    @params[:external_created_at].present? ? { external_created_at: @params[:external_created_at] } : {}
  end

  # 自动化规则ID
  def automation_rule_id
    @automation_rule.present? ? { content_attributes: { automation_rule_id: @automation_rule } } : {}
  end

  # 营销活动ID
  def campaign_id
    @params[:campaign_id].present? ? { additional_attributes: { campaign_id: @params[:campaign_id] } } : {}
  end

  # 模板参数
  def template_params
    @params[:template_params].present? ? { additional_attributes: { template_params: JSON.parse(@params[:template_params].to_json) } } : {}
  end

  # 消息发送者(用于机器人消息)
  def message_sender
    return if @params[:sender_type] != 'AgentBot'

    AgentBot.where(account_id: [nil, @conversation.account.id]).find_by(id: @params[:sender_id])
  end

  # 构建消息参数
  def message_params
    {
      account_id: @conversation.account_id,
      inbox_id: @conversation.inbox_id,
      message_type: message_type,
      content: @params[:content],
      private: @private,
      sender: sender,
      content_type: @params[:content_type],
      items: @items,
      in_reply_to: @in_reply_to,
      echo_id: @params[:echo_id],
      source_id: @params[:source_id]
    }.merge(external_created_at).merge(automation_rule_id).merge(campaign_id).merge(template_params)
  end
end
```

### 对话构建器实现

```ruby
class ConversationBuilder
  # 初始化对话构建器
  # @param params [Hash] 对话参数
  # @param contact_inbox [ContactInbox] 客户-收件箱关系
  def initialize(params:, contact_inbox:)
    @params = params
    @contact_inbox = contact_inbox
  end

  # 执行对话创建
  # @return [Conversation] 创建的对话或现有对话
  def perform
    look_up_exising_conversation || create_new_conversation
  end

  private

  # 查找现有对话(当收件箱设置为单一对话模式时)
  def look_up_exising_conversation
    return unless @contact_inbox.inbox.lock_to_single_conversation?

    @contact_inbox.conversations.last
  end

  # 创建新对话
  def create_new_conversation
    ::Conversation.create!(conversation_params)
  end

  # 构建对话参数
  def conversation_params
    additional_attributes = @params[:additional_attributes]&.permit! || {}
    custom_attributes = @params[:custom_attributes]&.permit! || {}
    status = @params[:status].present? ? { status: @params[:status] } : {}

    {
      account_id: @contact_inbox.inbox.account_id,
      inbox_id: @contact_inbox.inbox_id,
      contact_id: @contact_inbox.contact_id,
      contact_inbox_id: @contact_inbox.id,
      additional_attributes: additional_attributes,  # 额外属性(如来源页面等)
      custom_attributes: custom_attributes,          # 自定义属性
      snoozed_until: @params[:snoozed_until],       # 暂停到指定时间
      assignee_id: @params[:assignee_id],           # 分配的客服ID
      team_id: @params[:team_id]                    # 分配的团队ID
    }.merge(status)
  end
end
```

## 实时通信实现

### RoomChannel实现

```ruby
class RoomChannel < ApplicationCable::Channel
  # 订阅房间(账户级别)
  def subscribed
    ensure_stream
  end

  # 取消订阅
  def unsubscribed
    # 清理工作
  end

  # 更新在线状态
  def update_presence
    current_user.update!(availability: params[:availability])
    broadcast_presence_update
  end

  # 加入对话房间
  def typing_on
    broadcast_typing_indicator(true)
  end

  # 停止输入
  def typing_off
    broadcast_typing_indicator(false)
  end

  private

  # 确保流连接
  def ensure_stream
    stream_from room_token
  end

  # 房间标识符(基于账户ID和用户pubsub_token)
  def room_token
    "account_#{current_account.id}_user_#{current_user.pubsub_token}"
  end

  # 广播在线状态更新
  def broadcast_presence_update
    ActionCable.server.broadcast(
      room_token,
      {
        event: 'presence.update',
        data: {
          account_id: current_account.id,
          user_id: current_user.id,
          availability: current_user.availability
        }
      }
    )
  end

  # 广播输入状态指示器
  def broadcast_typing_indicator(is_typing)
    return unless params[:conversation_id].present?

    ActionCable.server.broadcast(
      "conversation_#{params[:conversation_id]}",
      {
        event: is_typing ? 'typing.on' : 'typing.off',
        data: {
          user_id: current_user.id,
          conversation_id: params[:conversation_id]
        }
      }
    )
  end
end
```

### 事件分发系统

```ruby
# 在模型中触发WebSocket事件
class Message < ApplicationRecord
  after_create_commit :dispatch_create_event
  after_update_commit :dispatch_update_event

  private

  def dispatch_create_event
    # 使用Rails配置的事件分发器
    Rails.configuration.dispatcher.dispatch(
      MESSAGE_CREATED,
      Time.zone.now,
      message: self,
      account: account
    )
  end

  def dispatch_update_event
    Rails.configuration.dispatcher.dispatch(
      MESSAGE_UPDATED,
      Time.zone.now,
      message: self,
      account: account,
      changed_attributes: previous_changes
    )
  end
end

# 事件监听器
class MessageListener < BaseListener
  # 处理消息创建事件
  def message_created(event)
    message = event.data[:message]

    # 广播到对话房间
    broadcast_to_conversation(message)

    # 广播到账户房间
    broadcast_to_account(message)

    # 触发通知
    trigger_notifications(message)
  end

  private

  def broadcast_to_conversation(message)
    ActionCable.server.broadcast(
      "conversation_#{message.conversation_id}",
      {
        event: 'message.created',
        data: message.push_event_data
      }
    )
  end

  def broadcast_to_account(message)
    ActionCable.server.broadcast(
      "account_#{message.account_id}",
      {
        event: 'message.created',
        data: {
          conversation: message.conversation.push_event_data,
          message: message.push_event_data
        }
      }
    )
  end
end
```

## 自动化规则引擎

### 自动化规则模型

```ruby
class AutomationRule < ApplicationRecord
  # 事件类型
  enum event_name: {
    conversation_created: 0,
    conversation_updated: 1,
    message_created: 2
  }

  # 规则状态
  enum rule_type: { condition: 0, action: 1 }

  # 关联关系
  belongs_to :account
  has_many_attached :files  # 支持附件动作

  # 条件和动作存储为JSON
  # conditions: [
  #   {
  #     attribute_key: "status",
  #     filter_operator: "equal_to",
  #     values: ["open"],
  #     query_operator: "and"
  #   }
  # ]
  # actions: [
  #   {
  #     action_name: "assign_agent",
  #     action_params: [1, 2, 3]  # agent IDs
  #   }
  # ]

  # 验证规则配置
  validates :name, presence: true
  validates :event_name, presence: true
  validates :conditions, presence: true
  validates :actions, presence: true

  # 检查规则是否匹配给定的对话
  def conditions_match?(conversation, options = {})
    return false unless active?

    # 使用条件服务检查所有条件
    AutomationRules::ConditionsFilterService.new(
      self,
      conversation,
      options
    ).perform.present?
  end

  # 执行规则动作
  def execute!(conversation, options = {})
    return unless conditions_match?(conversation, options)

    AutomationRules::ActionService.new(
      self,
      account,
      conversation
    ).perform
  end
end
```

### 条件过滤服务

```ruby
class AutomationRules::ConditionsFilterService
  # 初始化条件过滤服务
  def initialize(rule, conversation, options = {})
    @rule = rule
    @conversation = conversation
    @options = options
  end

  # 执行条件检查
  def perform
    return [] if @rule.conditions.blank?

    # 解析条件逻辑
    conversation_filter = ::Conversations::FilterService.new(
      filter_params,
      nil,
      @conversation.account
    )

    # 返回匹配的对话(如果当前对话匹配则返回数组，否则返回空数组)
    conversation_filter.perform[:conversations]
                      .where(id: @conversation.id)
  end

  private

  # 将自动化规则条件转换为过滤参数
  def filter_params
    {
      payload: [
        {
          attribute_key: 'status',
          filter_operator: 'equal_to',
          values: @rule.conditions.map { |condition| condition['values'] }.flatten,
          query_operator: 'and'
        }
      ].to_json
    }
  end
end
```

## 多渠道消息路由

### WhatsApp渠道实现

```ruby
class Channel::Whatsapp < ApplicationRecord
  # 渠道配置
  validates :phone_number, presence: true
  validates :provider, inclusion: { in: %w[default 360dialog whatsapp_cloud] }

  # 多态关联到收件箱
  has_one :inbox, as: :channel, dependent: :destroy

  # 支持的消息类型
  def supported_message_types
    %w[text image audio video document location sticker]
  end

  # 检查是否需要重新授权
  def reauthorization_required?
    authorization_error_count > 5
  end

  # 发送消息到WhatsApp
  def send_message(message)
    case provider
    when '360dialog'
      Whatsapp::Providers::Dialog360Service.new(whatsapp_channel: self).send_message(message)
    when 'whatsapp_cloud'
      Whatsapp::Providers::WhatsappCloudService.new(whatsapp_channel: self).send_message(message)
    else
      raise "Unsupported provider: #{provider}"
    end
  end

  # 处理入站消息
  def process_webhook(params)
    Whatsapp::IncomingMessageService.new(
      inbox: inbox,
      params: params
    ).perform
  end
end
```

### 消息发送服务

```ruby
class Whatsapp::Providers::WhatsappCloudService < Whatsapp::Providers::BaseService
  # 发送文本消息
  def send_text_message
    response = HTTParty.post(
      message_url,
      headers: headers,
      body: {
        messaging_product: 'whatsapp',
        to: contact_phone_number,
        type: 'text',
        text: { body: message.content }
      }.to_json
    )

    process_response(response)
  end

  # 发送媒体消息
  def send_attachment_message
    attachment = message.attachments.first

    response = HTTParty.post(
      message_url,
      headers: headers,
      body: {
        messaging_product: 'whatsapp',
        to: contact_phone_number,
        type: attachment.file_type,
        attachment.file_type.to_sym => {
          link: attachment.download_url,
          caption: message.content
        }
      }.to_json
    )

    process_response(response)
  end

  # 发送模板消息
  def send_template_message
    template_params = message.additional_attributes['template_params']

    response = HTTParty.post(
      message_url,
      headers: headers,
      body: {
        messaging_product: 'whatsapp',
        to: contact_phone_number,
        type: 'template',
        template: {
          name: template_params['name'],
          language: { code: template_params['language'] },
          components: build_template_components(template_params)
        }
      }.to_json
    )

    process_response(response)
  end

  private

  # 构建请求头
  def headers
    {
      'Authorization' => "Bearer #{whatsapp_channel.provider_config['api_key']}",
      'Content-Type' => 'application/json'
    }
  end

  # API端点URL
  def message_url
    "https://graph.facebook.com/v17.0/#{phone_number_id}/messages"
  end

  # 处理API响应
  def process_response(response)
    if response.success?
      # 更新消息状态为已发送
      message.update!(
        status: :sent,
        source_id: response.parsed_response.dig('messages', 0, 'id')
      )
    else
      # 记录发送失败
      message.update!(
        status: :failed,
        content_attributes: message.content_attributes.merge(
          external_error: response.parsed_response['error']
        )
      )
    end
  end
end
```

## 通知系统架构

### 新消息通知服务

```ruby
class Messages::NewMessageNotificationService
  # 初始化新消息通知服务
  # @param message [Message] 新创建的消息
  def initialize(message)
    @message = message
  end

  # 执行通知流程
  def perform
    return unless message.notifiable?  # 检查消息是否需要通知

    notify_conversation_assignee    # 通知对话分配的客服
    notify_participating_users     # 通知对话参与者
  end

  private

  delegate :conversation, :sender, :account, to: :message

  # 通知对话分配的客服
  def notify_conversation_assignee
    return if conversation.assignee.blank?
    return if already_notified?(conversation.assignee)  # 避免重复通知
    return if conversation.assignee == sender           # 不通知发送者自己

    NotificationBuilder.new(
      notification_type: 'assigned_conversation_new_message',
      user: conversation.assignee,
      account: account,
      primary_actor: message.conversation,
      secondary_actor: message
    ).perform
  end

  # 通知对话参与者
  def notify_participating_users
    participating_users = conversation.conversation_participants.map(&:user)
    participating_users -= [sender] if sender.is_a?(User)

    participating_users.uniq.each do |participant|
      next if already_notified?(participant)

      NotificationBuilder.new(
        notification_type: 'participating_conversation_new_message',
        user: participant,
        account: account,
        primary_actor: message.conversation,
        secondary_actor: message
      ).perform
    end
  end

  # 检查用户是否已被通知
  # (用户可能已通过@提及或分配通知，避免重复)
  def already_notified?(user)
    conversation.notifications.exists?(user: user, secondary_actor: message)
  end
end
```

### 通知构建器

```ruby
class NotificationBuilder
  # 初始化通知构建器
  # @param notification_type [String] 通知类型
  # @param user [User] 接收通知的用户
  # @param account [Account] 账户
  # @param primary_actor [Object] 主要关联对象
  # @param secondary_actor [Object] 次要关联对象
  def initialize(notification_type:, user:, account:, primary_actor:, secondary_actor: nil)
    @notification_type = notification_type
    @user = user
    @account = account
    @primary_actor = primary_actor
    @secondary_actor = secondary_actor
  end

  # 执行通知创建
  def perform
    return unless should_send_notification?

    notification = create_notification
    send_push_notification(notification)
    send_email_notification(notification)
  end

  private

  # 检查是否应该发送通知
  def should_send_notification?
    return false if @user.blank?
    return false if @account.blank?
    return false unless notification_setting_enabled?

    true
  end

  # 检查通知设置是否启用
  def notification_setting_enabled?
    notification_setting = @user.notification_settings.find_by(account: @account)
    return true if notification_setting.blank?  # 默认启用

    notification_setting.public_send("push_#{@notification_type}?")
  end

  # 创建通知记录
  def create_notification
    Notification.create!(
      notification_type: @notification_type,
      user: @user,
      account: @account,
      primary_actor: @primary_actor,
      secondary_actor: @secondary_actor
    )
  end

  # 发送推送通知
  def send_push_notification(notification)
    Notification::PushNotificationService.new(notification: notification).perform
  end

  # 发送邮件通知
  def send_email_notification(notification)
    Notification::EmailNotificationService.new(notification: notification).perform
  end
end
```

### 邮件通知服务

```ruby
class Notification::EmailNotificationService
  # 初始化邮件通知服务
  # @param notification [Notification] 通知对象
  def initialize(notification:)
    @notification = notification
  end

  # 执行邮件发送
  def perform
    # 如果用户已读推送通知则不发送邮件
    return if @notification.read_at.present?
    # 如果用户未确认邮箱则不发送邮件
    return if @notification.user.confirmed_at.nil?
    return unless user_subscribed_to_notification?

    # 发送邮件通知
    AgentNotifications::ConversationNotificationsMailer
      .with(account: @notification.account)
      .public_send(@notification.notification_type.to_s,
                   @notification.primary_actor,
                   @notification.user,
                   @notification.secondary_actor)
      .deliver_later
  end

  private

  # 检查用户是否订阅了该类型的邮件通知
  def user_subscribed_to_notification?
    notification_setting = @notification.user.notification_settings
                                            .find_by(account_id: @notification.account.id)
    return true if notification_setting.blank?  # 默认启用

    notification_setting.public_send("email_#{@notification.notification_type}?")
  end
end
```

### 推送通知服务

```ruby
class Notification::PushNotificationService
  # 初始化推送通知服务
  # @param notification [Notification] 通知对象
  def initialize(notification:)
    @notification = notification
  end

  # 执行推送通知发送
  def perform
    return unless user_subscribed_to_push_notifications?

    send_fcm_notification if fcm_tokens.present?
    send_web_push_notification if web_push_subscriptions.present?
  end

  private

  # 检查用户是否订阅了推送通知
  def user_subscribed_to_push_notifications?
    notification_setting = @notification.user.notification_settings
                                            .find_by(account_id: @notification.account.id)
    return true if notification_setting.blank?

    notification_setting.public_send("push_#{@notification.notification_type}?")
  end

  # 获取用户的FCM令牌
  def fcm_tokens
    @fcm_tokens ||= @notification.user.notification_subscriptions
                                      .where(subscription_type: 'fcm')
                                      .pluck(:subscription_attributes)
                                      .map { |attr| attr['push_token'] }
                                      .compact
  end

  # 获取Web推送订阅
  def web_push_subscriptions
    @web_push_subscriptions ||= @notification.user.notification_subscriptions
                                                  .where(subscription_type: 'web_push')
                                                  .pluck(:subscription_attributes)
  end

  # 发送FCM推送通知
  def send_fcm_notification
    fcm = FCM.new(Rails.application.secrets.fcm_server_key)

    fcm_tokens.each do |token|
      response = fcm.send(
        [token],
        {
          notification: {
            title: notification_title,
            body: notification_body,
            icon: notification_icon
          },
          data: notification_data
        }
      )

      handle_fcm_response(response, token)
    end
  end

  # 发送Web推送通知
  def send_web_push_notification
    web_push_subscriptions.each do |subscription|
      message = {
        title: notification_title,
        body: notification_body,
        icon: notification_icon,
        data: notification_data
      }

      WebPush.payload_send(
        message: message.to_json,
        endpoint: subscription['endpoint'],
        p256dh: subscription['keys']['p256dh'],
        auth: subscription['keys']['auth'],
        vapid: {
          subject: 'mailto:support@chatwoot.com',
          public_key: Rails.application.secrets.vapid_public_key,
          private_key: Rails.application.secrets.vapid_private_key
        }
      )
    rescue WebPush::InvalidSubscription
      # 清理无效的订阅
      cleanup_invalid_subscription(subscription)
    end
  end

  # 构建通知标题
  def notification_title
    case @notification.notification_type
    when 'assigned_conversation_new_message'
      "New message in conversation ##{@notification.primary_actor.display_id}"
    when 'participating_conversation_new_message'
      "New message in conversation you're participating"
    when 'conversation_mention'
      "You were mentioned in a conversation"
    else
      'New notification'
    end
  end

  # 构建通知内容
  def notification_body
    case @notification.notification_type
    when 'assigned_conversation_new_message', 'participating_conversation_new_message'
      @notification.secondary_actor&.content&.truncate(100) || 'New message received'
    when 'conversation_mention'
      "#{@notification.secondary_actor.sender.name} mentioned you"
    else
      'You have a new notification'
    end
  end

  # 通知图标
  def notification_icon
    '/favicon.ico'
  end

  # 通知数据
  def notification_data
    {
      notification_id: @notification.id,
      conversation_id: @notification.primary_actor.id,
      account_id: @notification.account.id
    }
  end

  # 处理FCM响应
  def handle_fcm_response(response, token)
    if response[:failure] > 0
      Rails.logger.error "FCM notification failed for token #{token}: #{response[:results]}"
      # 清理无效的令牌
      cleanup_invalid_token(token) if response[:results].first[:error] == 'InvalidRegistration'
    end
  end

  # 清理无效的FCM令牌
  def cleanup_invalid_token(token)
    @notification.user.notification_subscriptions
                     .where(subscription_type: 'fcm')
                     .where("subscription_attributes ->> 'push_token' = ?", token)
                     .destroy_all
  end

  # 清理无效的Web推送订阅
  def cleanup_invalid_subscription(subscription)
    @notification.user.notification_subscriptions
                     .where(subscription_type: 'web_push')
                     .where(subscription_attributes: subscription)
                     .destroy_all
  end
end
```

## 总结

这份技术实现详解文档深入分析了Chatwoot项目的核心技术实现，包括：

1. **服务层架构**: 展示了如何使用服务对象模式组织复杂的业务逻辑
2. **消息处理流程**: 详细说明了消息创建、处理和分发的完整流程
3. **实时通信**: 基于ActionCable的WebSocket实现和事件广播机制
4. **自动化引擎**: 规则匹配和动作执行的完整实现
5. **多渠道路由**: 统一的消息路由和渠道适配器模式
6. **通知系统**: 多渠道通知的完整实现，包括推送、邮件等

通过学习这些技术实现，开发者可以深入理解：

- 如何设计可扩展的服务层架构
- 如何实现复杂的业务逻辑处理流程
- 如何构建实时通信系统
- 如何设计灵活的规则引擎
- 如何实现多渠道消息路由
- 如何构建完整的通知系统

这些实现展示了现代Rails应用的最佳实践，对于构建类似的复杂业务系统具有很高的参考价值。
