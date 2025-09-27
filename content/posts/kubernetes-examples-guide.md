# Kubernetes 框架使用示例指南

## 📚 文档概述

本文档提供了 Kubernetes 框架的详细使用示例，涵盖从基础概念到高级特性的完整实战案例。通过具体的 YAML 配置和代码示例，帮助用户快速上手 Kubernetes 的各种功能特性。

## 🚀 快速入门示例

### 1.1 第一个 Pod

```yaml
# 最简单的 Pod 示例
apiVersion: v1
kind: Pod
metadata:
  name: hello-world
  labels:
    app: hello-world
spec:
  containers:
  - name: hello
    image: nginx:1.21
    ports:
    - containerPort: 80
```

```bash
# 部署和管理 Pod
kubectl apply -f hello-world-pod.yaml
kubectl get pods
kubectl describe pod hello-world
kubectl logs hello-world
kubectl delete pod hello-world
```

### 1.2 第一个 Deployment

```yaml
# 基础 Deployment 示例
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 256Mi
```

```bash
# 部署和管理 Deployment
kubectl apply -f nginx-deployment.yaml
kubectl get deployments
kubectl get pods -l app=nginx
kubectl scale deployment nginx-deployment --replicas=5
kubectl rollout status deployment nginx-deployment
kubectl rollout history deployment nginx-deployment
```

### 1.3 第一个 Service

```yaml
# ClusterIP Service 示例
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
  labels:
    app: nginx
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 80
    protocol: TCP
  selector:
    app: nginx

---
# NodePort Service 示例
apiVersion: v1
kind: Service
metadata:
  name: nginx-nodeport
  labels:
    app: nginx
spec:
  type: NodePort
  ports:
  - port: 80
    targetPort: 80
    nodePort: 30080
    protocol: TCP
  selector:
    app: nginx

---
# LoadBalancer Service 示例
apiVersion: v1
kind: Service
metadata:
  name: nginx-loadbalancer
  labels:
    app: nginx
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 80
    protocol: TCP
  selector:
    app: nginx
```

```bash
# 测试服务连通性
kubectl apply -f nginx-service.yaml
kubectl get services
kubectl describe service nginx-service

# 端口转发测试
kubectl port-forward service/nginx-service 8080:80

# 在集群内测试
kubectl run test-pod --image=busybox --rm -it --restart=Never -- wget -qO- nginx-service
```

## 🔧 配置管理示例

### 2.1 ConfigMap 使用示例

```yaml
# ConfigMap 创建
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  # 键值对配置
  database_url: "mysql://db.example.com:3306/myapp"
  debug_mode: "true"
  max_connections: "100"
  
  # 配置文件
  app.properties: |
    server.port=8080
    server.servlet.context-path=/api
    spring.datasource.url=jdbc:mysql://db.example.com:3306/myapp
    spring.datasource.username=appuser
    logging.level.com.example=INFO
  
  # Nginx 配置
  nginx.conf: |
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://backend:8080;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /health {
            return 200 "OK";
        }
    }

---
# 使用 ConfigMap 的 Pod
apiVersion: v1
kind: Pod
metadata:
  name: app-with-config
spec:
  containers:
  - name: app
    image: myapp:latest
    
    # 方式1：环境变量
    env:
    - name: DATABASE_URL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: database_url
    - name: DEBUG_MODE
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: debug_mode
    
    # 方式2：环境变量（批量导入）
    envFrom:
    - configMapRef:
        name: app-config
    
    # 方式3：文件挂载
    volumeMounts:
    - name: config-volume
      mountPath: /etc/config
    - name: nginx-config
      mountPath: /etc/nginx/nginx.conf
      subPath: nginx.conf
  
  volumes:
  - name: config-volume
    configMap:
      name: app-config
  - name: nginx-config
    configMap:
      name: app-config
      items:
      - key: nginx.conf
        path: nginx.conf
```

```bash
# ConfigMap 管理命令
# 从文件创建
kubectl create configmap app-config --from-file=config/
kubectl create configmap app-config --from-file=app.properties

# 从字面值创建
kubectl create configmap app-config --from-literal=key1=value1 --from-literal=key2=value2

# 查看和编辑
kubectl get configmaps
kubectl describe configmap app-config
kubectl edit configmap app-config

# 更新配置后重启 Pod
kubectl rollout restart deployment/app-deployment
```

### 2.2 Secret 使用示例

```yaml
# 通用 Secret
apiVersion: v1
kind: Secret
metadata:
  name: app-secret
type: Opaque
data:
  # Base64 编码的值
  username: YWRtaW4=  # admin
  password: cGFzc3dvcmQxMjM=  # password123
  api-key: YWJjZGVmZ2hpams=  # abcdefghijk

---
# TLS Secret
apiVersion: v1
kind: Secret
metadata:
  name: tls-secret
type: kubernetes.io/tls
data:
  tls.crt: |
    LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0t...
  tls.key: |
    LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0t...

---
# Docker Registry Secret
apiVersion: v1
kind: Secret
metadata:
  name: registry-secret
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: |
    eyJhdXRocyI6eyJteXJlZ2lzdHJ5LmNvbSI6eyJ1c2VybmFtZSI6InVzZXIiLCJwYXNzd29yZCI6InBhc3MiLCJhdXRoIjoiZFhObGNqcHdZWE56In19fQ==

---
# 使用 Secret 的 Pod
apiVersion: v1
kind: Pod
metadata:
  name: app-with-secret
spec:
  containers:
  - name: app
    image: myregistry.com/myapp:latest
    
    # 环境变量方式
    env:
    - name: DB_USERNAME
      valueFrom:
        secretKeyRef:
          name: app-secret
          key: username
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: app-secret
          key: password
    
    # 文件挂载方式
    volumeMounts:
    - name: secret-volume
      mountPath: /etc/secrets
      readOnly: true
    - name: tls-volume
      mountPath: /etc/tls
      readOnly: true
  
  # 镜像拉取密钥
  imagePullSecrets:
  - name: registry-secret
  
  volumes:
  - name: secret-volume
    secret:
      secretName: app-secret
  - name: tls-volume
    secret:
      secretName: tls-secret
```

```bash
# Secret 管理命令
# 创建通用 Secret
kubectl create secret generic app-secret \
  --from-literal=username=admin \
  --from-literal=password=password123

# 创建 TLS Secret
kubectl create secret tls tls-secret \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key

# 创建 Docker Registry Secret
kubectl create secret docker-registry registry-secret \
  --docker-server=myregistry.com \
  --docker-username=user \
  --docker-password=pass \
  --docker-email=user@example.com

# 查看 Secret（不显示值）
kubectl get secrets
kubectl describe secret app-secret

# 查看 Secret 内容
kubectl get secret app-secret -o yaml
kubectl get secret app-secret -o jsonpath='{.data.username}' | base64 -d
```

## 💾 存储管理示例

### 3.1 Volume 使用示例

```yaml
# EmptyDir Volume
apiVersion: v1
kind: Pod
metadata:
  name: pod-with-emptydir
spec:
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - name: cache-volume
      mountPath: /cache
  - name: sidecar
    image: busybox
    command: ['sh', '-c', 'while true; do echo $(date) >> /cache/log.txt; sleep 10; done']
    volumeMounts:
    - name: cache-volume
      mountPath: /cache
  volumes:
  - name: cache-volume
    emptyDir:
      sizeLimit: 1Gi

---
# HostPath Volume
apiVersion: v1
kind: Pod
metadata:
  name: pod-with-hostpath
spec:
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - name: host-volume
      mountPath: /host-data
  volumes:
  - name: host-volume
    hostPath:
      path: /data
      type: DirectoryOrCreate

---
# NFS Volume
apiVersion: v1
kind: Pod
metadata:
  name: pod-with-nfs
spec:
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - name: nfs-volume
      mountPath: /nfs-data
  volumes:
  - name: nfs-volume
    nfs:
      server: nfs-server.example.com
      path: /exported/path
```

### 3.2 PersistentVolume 和 PersistentVolumeClaim

```yaml
# StorageClass 定义
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
reclaimPolicy: Retain
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer

---
# PersistentVolume 手动创建
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-example
spec:
  capacity:
    storage: 10Gi
  accessModes:
  - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: manual
  hostPath:
    path: /data/pv-example

---
# PersistentVolumeClaim
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: app-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: fast-ssd

---
# 使用 PVC 的 Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-with-storage
spec:
  replicas: 1
  selector:
    matchLabels:
      app: app-with-storage
  template:
    metadata:
      labels:
        app: app-with-storage
    spec:
      containers:
      - name: app
        image: postgres:13
        env:
        - name: POSTGRES_DB
          value: myapp
        - name: POSTGRES_USER
          value: user
        - name: POSTGRES_PASSWORD
          value: password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: app-pvc
```

```bash
# 存储管理命令
kubectl get storageclass
kubectl get pv
kubectl get pvc
kubectl describe pvc app-pvc

# 扩展 PVC（如果 StorageClass 支持）
kubectl patch pvc app-pvc -p '{"spec":{"resources":{"requests":{"storage":"10Gi"}}}}'

# 查看存储使用情况
kubectl exec -it deployment/app-with-storage -- df -h
```

## 🔄 工作负载示例

### 4.1 StatefulSet 示例

```yaml
# StatefulSet 示例（MySQL 集群）
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql-cluster
spec:
  serviceName: mysql-headless
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - name: mysql
        image: mysql:8.0
        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: root-password
        - name: MYSQL_REPLICATION_USER
          value: replicator
        - name: MYSQL_REPLICATION_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: replication-password
        ports:
        - containerPort: 3306
          name: mysql
        volumeMounts:
        - name: mysql-data
          mountPath: /var/lib/mysql
        - name: mysql-config
          mountPath: /etc/mysql/conf.d
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1
            memory: 2Gi
        livenessProbe:
          exec:
            command:
            - mysqladmin
            - ping
            - -h
            - localhost
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - mysql
            - -h
            - localhost
            - -e
            - "SELECT 1"
          initialDelaySeconds: 5
          periodSeconds: 2
      volumes:
      - name: mysql-config
        configMap:
          name: mysql-config
  volumeClaimTemplates:
  - metadata:
      name: mysql-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
      storageClassName: fast-ssd

---
# Headless Service for StatefulSet
apiVersion: v1
kind: Service
metadata:
  name: mysql-headless
spec:
  clusterIP: None
  selector:
    app: mysql
  ports:
  - port: 3306
    name: mysql

---
# MySQL 配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: mysql-config
data:
  my.cnf: |
    [mysqld]
    server-id=1
    log-bin=mysql-bin
    binlog-format=ROW
    gtid-mode=ON
    enforce-gtid-consistency=ON
    master-info-repository=TABLE
    relay-log-info-repository=TABLE
    binlog-checksum=NONE
    log-slave-updates=ON
    log-bin-trust-function-creators=ON
    slave-preserve-commit-order=ON

---
# MySQL Secret
apiVersion: v1
kind: Secret
metadata:
  name: mysql-secret
type: Opaque
data:
  root-password: cm9vdHBhc3N3b3Jk  # rootpassword
  replication-password: cmVwbGljYXRvcg==  # replicator
```

### 4.2 DaemonSet 示例

```yaml
# DaemonSet 示例（日志收集器）
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd-logger
  namespace: kube-system
  labels:
    app: fluentd-logger
spec:
  selector:
    matchLabels:
      app: fluentd-logger
  template:
    metadata:
      labels:
        app: fluentd-logger
    spec:
      serviceAccountName: fluentd
      tolerations:
      # 允许在主节点运行
      - key: node-role.kubernetes.io/control-plane
        operator: Exists
        effect: NoSchedule
      - key: node-role.kubernetes.io/master
        operator: Exists
        effect: NoSchedule
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch.logging.svc.cluster.local"
        - name: FLUENT_ELASTICSEARCH_PORT
          value: "9200"
        - name: FLUENT_ELASTICSEARCH_SCHEME
          value: "http"
        - name: FLUENTD_SYSTEMD_CONF
          value: disable
        resources:
          limits:
            memory: 512Mi
          requests:
            cpu: 100m
            memory: 200Mi
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
        - name: fluentd-config
          mountPath: /fluentd/etc
      terminationGracePeriodSeconds: 30
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
      - name: fluentd-config
        configMap:
          name: fluentd-config

---
# ServiceAccount for DaemonSet
apiVersion: v1
kind: ServiceAccount
metadata:
  name: fluentd
  namespace: kube-system

---
# ClusterRole for log collection
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: fluentd
rules:
- apiGroups: [""]
  resources: ["pods", "namespaces"]
  verbs: ["get", "list", "watch"]

---
# ClusterRoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: fluentd
roleRef:
  kind: ClusterRole
  name: fluentd
  apiGroup: rbac.authorization.k8s.io
subjects:
- kind: ServiceAccount
  name: fluentd
  namespace: kube-system
```

### 4.3 Job 和 CronJob 示例

```yaml
# Job 示例（数据库迁移）
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration
spec:
  template:
    metadata:
      labels:
        app: db-migration
    spec:
      restartPolicy: Never
      containers:
      - name: migration
        image: migrate/migrate:latest
        command:
        - migrate
        - -path
        - /migrations
        - -database
        - postgres://user:password@db:5432/myapp?sslmode=disable
        - up
        volumeMounts:
        - name: migration-scripts
          mountPath: /migrations
        env:
        - name: DB_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
      volumes:
      - name: migration-scripts
        configMap:
          name: migration-scripts
  backoffLimit: 3
  activeDeadlineSeconds: 300

---
# CronJob 示例（定期备份）
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
spec:
  schedule: "0 2 * * *"  # 每天凌晨 2 点
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: backup
            image: postgres:13
            command:
            - /bin/bash
            - -c
            - |
              BACKUP_FILE="/backup/backup-$(date +%Y%m%d-%H%M%S).sql"
              pg_dump $DATABASE_URL > $BACKUP_FILE
              echo "Backup completed: $BACKUP_FILE"
              
              # 清理 7 天前的备份
              find /backup -name "backup-*.sql" -mtime +7 -delete
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: url
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
  concurrencyPolicy: Forbid

---
# 并行 Job 示例
apiVersion: batch/v1
kind: Job
metadata:
  name: parallel-processing
spec:
  parallelism: 5  # 并行运行 5 个 Pod
  completions: 20  # 总共需要完成 20 个任务
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: worker
        image: busybox
        command:
        - /bin/sh
        - -c
        - |
          echo "Processing job $JOB_COMPLETION_INDEX"
          sleep $((RANDOM % 60 + 30))  # 模拟 30-90 秒的工作
          echo "Job $JOB_COMPLETION_INDEX completed"
        env:
        - name: JOB_COMPLETION_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
```

## 🌐 网络和服务发现

### 5.1 Ingress 示例

```yaml
# Ingress Controller (Nginx)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-ingress-controller
  namespace: ingress-nginx
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nginx-ingress-controller
  template:
    metadata:
      labels:
        app: nginx-ingress-controller
    spec:
      serviceAccountName: nginx-ingress-serviceaccount
      containers:
      - name: nginx-ingress-controller
        image: k8s.gcr.io/ingress-nginx/controller:v1.8.1
        args:
        - /nginx-ingress-controller
        - --configmap=$(POD_NAMESPACE)/nginx-configuration
        - --tcp-services-configmap=$(POD_NAMESPACE)/tcp-services
        - --udp-services-configmap=$(POD_NAMESPACE)/udp-services
        - --publish-service=$(POD_NAMESPACE)/ingress-nginx
        - --annotations-prefix=nginx.ingress.kubernetes.io
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        ports:
        - name: http
          containerPort: 80
        - name: https
          containerPort: 443
        resources:
          requests:
            cpu: 100m
            memory: 90Mi

---
# Ingress 资源示例
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-ingress
  annotations:
    # Nginx 特定注解
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/use-regex: "true"
    
    # 限流
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    
    # 负载均衡
    nginx.ingress.kubernetes.io/load-balance: "round_robin"
    
    # 会话亲和性
    nginx.ingress.kubernetes.io/affinity: "cookie"
    nginx.ingress.kubernetes.io/session-cookie-name: "route"
    nginx.ingress.kubernetes.io/session-cookie-expires: "86400"
    
    # CORS
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"
    nginx.ingress.kubernetes.io/cors-allow-headers: "DNT,X-CustomHeader,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Authorization"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - example.com
    - api.example.com
    secretName: tls-secret
  rules:
  # 主站点
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80
  
  # API 服务
  - host: api.example.com
    http:
      paths:
      - path: /v1
        pathType: Prefix
        backend:
          service:
            name: api-v1-service
            port:
              number: 8080
      - path: /v2
        pathType: Prefix
        backend:
          service:
            name: api-v2-service
            port:
              number: 8080
  
  # 基于路径的路由
  - host: app.example.com
    http:
      paths:
      - path: /admin
        pathType: Prefix
        backend:
          service:
            name: admin-service
            port:
              number: 3000
      - path: /user
        pathType: Prefix
        backend:
          service:
            name: user-service
            port:
              number: 3000
      - path: /static
        pathType: Prefix
        backend:
          service:
            name: static-service
            port:
              number: 80

---
# 多个 Ingress 类示例
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: traefik-ingress
  annotations:
    traefik.ingress.kubernetes.io/router.middlewares: default-auth@kubernetescrd
spec:
  ingressClassName: traefik
  rules:
  - host: traefik.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80
```

### 5.2 NetworkPolicy 示例

```yaml
# 默认拒绝策略
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

---
# 允许前端访问后端
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      tier: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          tier: frontend
    ports:
    - protocol: TCP
      port: 8080

---
# 允许后端访问数据库
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-to-database
  namespace: production
spec:
  podSelector:
    matchLabels:
      tier: backend
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          tier: database
    ports:
    - protocol: TCP
      port: 5432

---
# 跨命名空间访问
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cross-namespace-access
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: web-app
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
      podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 8080

---
# 允许 DNS 和外部访问
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns-and-external
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: web-app
  policyTypes:
  - Egress
  egress:
  # 允许 DNS 解析
  - to: []
    ports:
    - protocol: UDP
      port: 53
  # 允许 HTTPS 出站
  - to: []
    ports:
    - protocol: TCP
      port: 443
  # 允许访问特定外部服务
  - to:
    - namespaceSelector: {}
      podSelector:
        matchLabels:
          app: external-api
    ports:
    - protocol: TCP
      port: 8080
```

## 🔐 安全和权限管理

### 6.1 RBAC 完整示例

```yaml
# 命名空间
apiVersion: v1
kind: Namespace
metadata:
  name: development

---
# ServiceAccount
apiVersion: v1
kind: ServiceAccount
metadata:
  name: developer
  namespace: development

---
# Role - 开发者权限
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: development
  name: developer-role
rules:
# Pod 管理权限
- apiGroups: [""]
  resources: ["pods", "pods/log", "pods/exec"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# Service 管理权限
- apiGroups: [""]
  resources: ["services"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# ConfigMap 和 Secret 权限
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
# Deployment 权限
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# Ingress 权限
- apiGroups: ["networking.k8s.io"]
  resources: ["ingresses"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# 事件查看权限
- apiGroups: [""]
  resources: ["events"]
  verbs: ["get", "list", "watch"]

---
# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: developer-binding
  namespace: development
subjects:
- kind: ServiceAccount
  name: developer
  namespace: development
- kind: User
  name: john.doe@example.com
  apiGroup: rbac.authorization.k8s.io
- kind: Group
  name: developers
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: developer-role
  apiGroup: rbac.authorization.k8s.io

---
# ClusterRole - 只读权限
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: readonly-cluster
rules:
# 查看节点信息
- apiGroups: [""]
  resources: ["nodes", "nodes/status"]
  verbs: ["get", "list", "watch"]
# 查看命名空间
- apiGroups: [""]
  resources: ["namespaces"]
  verbs: ["get", "list", "watch"]
# 查看存储类
- apiGroups: ["storage.k8s.io"]
  resources: ["storageclasses"]
  verbs: ["get", "list", "watch"]
# 查看 CRD
- apiGroups: ["apiextensions.k8s.io"]
  resources: ["customresourcedefinitions"]
  verbs: ["get", "list", "watch"]

---
# ClusterRoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: readonly-binding
subjects:
- kind: Group
  name: viewers
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: readonly-cluster
  apiGroup: rbac.authorization.k8s.io

---
# 管理员 ClusterRole
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: admin-cluster
rules:
# 完全权限（除了一些系统资源）
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
# 排除一些敏感操作
- apiGroups: [""]
  resources: ["nodes"]
  verbs: ["get", "list", "watch"]
  resourceNames: []

---
# 应用专用 ServiceAccount
apiVersion: v1
kind: ServiceAccount
metadata:
  name: monitoring-sa
  namespace: monitoring

---
# 监控应用权限
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: monitoring-reader
rules:
- apiGroups: [""]
  resources: ["nodes", "nodes/metrics", "services", "endpoints", "pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets", "daemonsets", "statefulsets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["batch"]
  resources: ["jobs", "cronjobs"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: monitoring-binding
subjects:
- kind: ServiceAccount
  name: monitoring-sa
  namespace: monitoring
roleRef:
  kind: ClusterRole
  name: monitoring-reader
  apiGroup: rbac.authorization.k8s.io
```

### 6.2 Pod Security Context 示例

```yaml
# 安全的 Pod 配置
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
  annotations:
    seccomp.security.alpha.kubernetes.io/pod: runtime/default
spec:
  # Pod 级别安全上下文
  securityContext:
    # 运行为非 root 用户
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    
    # 设置 seccomp 配置文件
    seccompProfile:
      type: RuntimeDefault
    
    # 设置 SELinux 选项
    seLinuxOptions:
      level: "s0:c123,c456"
    
    # 设置 sysctl 参数
    sysctls:
    - name: net.core.somaxconn
      value: "1024"
  
  containers:
  - name: app
    image: nginx:1.21
    
    # 容器级别安全上下文
    securityContext:
      # 禁止特权提升
      allowPrivilegeEscalation: false
      
      # 只读根文件系统
      readOnlyRootFilesystem: true
      
      # 不以特权模式运行
      privileged: false
      
      # 删除所有 capabilities
      capabilities:
        drop:
        - ALL
        add:
        - NET_BIND_SERVICE  # 只添加必要的 capability
      
      # 运行为特定用户
      runAsNonRoot: true
      runAsUser: 1001
      runAsGroup: 1001
    
    # 资源限制
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
      limits:
        cpu: 200m
        memory: 256Mi
    
    # 挂载临时文件系统
    volumeMounts:
    - name: tmp-volume
      mountPath: /tmp
    - name: var-cache-nginx
      mountPath: /var/cache/nginx
    - name: var-run
      mountPath: /var/run
  
  volumes:
  - name: tmp-volume
    emptyDir: {}
  - name: var-cache-nginx
    emptyDir: {}
  - name: var-run
    emptyDir: {}

---
# Pod Security Policy (已弃用，使用 Pod Security Standards)
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
  - ALL
  volumes:
  - 'configMap'
  - 'emptyDir'
  - 'projected'
  - 'secret'
  - 'downwardAPI'
  - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true

---
# 使用 Pod Security Standards 的命名空间
apiVersion: v1
kind: Namespace
metadata:
  name: secure-namespace
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

## 📊 监控和调试示例

### 7.1 应用监控配置

```yaml
# 带监控的应用 Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: monitored-app
  labels:
    app: monitored-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: monitored-app
  template:
    metadata:
      labels:
        app: monitored-app
      annotations:
        # Prometheus 抓取配置
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: app
        image: myapp:latest
        ports:
        - name: http
          containerPort: 8080
        - name: metrics
          containerPort: 9090
        
        # 健康检查
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        
        # 启动探针
        startupProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 30
        
        # 环境变量
        env:
        - name: METRICS_PORT
          value: "9090"
        - name: LOG_LEVEL
          value: "info"
        
        # 资源配置
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi

---
# ServiceMonitor (Prometheus Operator)
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: monitored-app
  labels:
    app: monitored-app
spec:
  selector:
    matchLabels:
      app: monitored-app
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics

---
# PrometheusRule (告警规则)
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: monitored-app-rules
  labels:
    app: monitored-app
spec:
  groups:
  - name: monitored-app.rules
    rules:
    - alert: HighErrorRate
      expr: |
        sum(rate(http_requests_total{job="monitored-app",status=~"5.."}[5m])) /
        sum(rate(http_requests_total{job="monitored-app"}[5m])) > 0.05
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "应用错误率过高"
        description: "应用 {{ $labels.job }} 错误率为 {{ $value | humanizePercentage }}"
    
    - alert: HighLatency
      expr: |
        histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="monitored-app"}[5m])) by (le)) > 1
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "应用响应时间过长"
        description: "应用 {{ $labels.job }} 95% 响应时间为 {{ $value }}s"
```

### 7.2 调试和故障排查

```bash
#!/bin/bash
# Kubernetes 调试脚本

# 1. 集群状态检查
echo "=== 集群状态检查 ==="
kubectl cluster-info
kubectl get nodes -o wide
kubectl get componentstatuses

# 2. Pod 状态检查
echo "=== Pod 状态检查 ==="
kubectl get pods --all-namespaces -o wide
kubectl get pods --field-selector=status.phase=Failed --all-namespaces
kubectl get pods --field-selector=status.phase=Pending --all-namespaces

# 3. 事件查看
echo "=== 集群事件 ==="
kubectl get events --sort-by=.metadata.creationTimestamp --all-namespaces

# 4. 资源使用情况
echo "=== 资源使用情况 ==="
kubectl top nodes
kubectl top pods --all-namespaces

# 5. 网络连接测试
echo "=== 网络连接测试 ==="
kubectl run test-pod --image=busybox --rm -it --restart=Never -- nslookup kubernetes.default

# 6. 存储状态检查
echo "=== 存储状态 ==="
kubectl get pv,pvc --all-namespaces
kubectl get storageclass

# 7. 服务和端点检查
echo "=== 服务和端点 ==="
kubectl get svc,endpoints --all-namespaces

# 8. Ingress 状态
echo "=== Ingress 状态 ==="
kubectl get ingress --all-namespaces

# 9. 特定 Pod 详细信息
debug_pod() {
    local pod_name=$1
    local namespace=${2:-default}
    
    echo "=== 调试 Pod: $namespace/$pod_name ==="
    kubectl describe pod $pod_name -n $namespace
    kubectl logs $pod_name -n $namespace --previous
    kubectl logs $pod_name -n $namespace
}

# 10. 进入 Pod 进行调试
debug_exec() {
    local pod_name=$1
    local namespace=${2:-default}
    local container=${3:-}
    
    if [ -n "$container" ]; then
        kubectl exec -it $pod_name -n $namespace -c $container -- /bin/sh
    else
        kubectl exec -it $pod_name -n $namespace -- /bin/sh
    fi
}

# 11. 端口转发调试
port_forward() {
    local service_name=$1
    local local_port=$2
    local remote_port=$3
    local namespace=${4:-default}
    
    kubectl port-forward service/$service_name $local_port:$remote_port -n $namespace
}

# 12. 网络策略测试
test_network_policy() {
    local source_pod=$1
    local target_service=$2
    local namespace=${3:-default}
    
    kubectl exec -it $source_pod -n $namespace -- wget -qO- --timeout=5 $target_service
}

# 使用示例
# debug_pod "my-pod" "default"
# debug_exec "my-pod" "default" "my-container"
# port_forward "my-service" 8080 80 "default"
# test_network_policy "test-pod" "target-service" "default"
```

## 📚 总结

### 框架使用要点

1. **渐进式学习**：从基础的 Pod、Service 开始，逐步掌握复杂的工作负载
2. **最佳实践**：始终遵循安全、监控、资源管理的最佳实践
3. **声明式管理**：使用 YAML 文件进行声明式配置管理
4. **标签和选择器**：合理使用标签进行资源组织和选择
5. **健康检查**：为所有应用配置适当的健康检查

### 常用命令总结

```bash
# 资源管理
kubectl apply -f <file>
kubectl delete -f <file>
kubectl get <resource>
kubectl describe <resource> <name>
kubectl edit <resource> <name>

# 调试和故障排查
kubectl logs <pod> -f
kubectl exec -it <pod> -- /bin/sh
kubectl port-forward <pod> <local-port>:<remote-port>
kubectl top nodes/pods

# 扩缩容
kubectl scale deployment <name> --replicas=<count>
kubectl autoscale deployment <name> --min=<min> --max=<max> --cpu-percent=<percent>

# 滚动更新
kubectl set image deployment/<name> <container>=<image>
kubectl rollout status deployment/<name>
kubectl rollout undo deployment/<name>
```

通过这些示例和最佳实践，您可以快速掌握 Kubernetes 的各种功能特性，并在生产环境中安全、高效地使用 Kubernetes。

---

**文档版本**: v1.0  
**最后更新**: 2025年09月27日  
**适用版本**: Kubernetes 1.29+
