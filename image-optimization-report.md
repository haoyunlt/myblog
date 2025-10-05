# 图片优化报告

生成时间：Sun Oct  5 23:50:12 CST 2025

## 优化策略

### JPEG优化
- 质量设置：85%
- 启用渐进式加载
- 移除EXIF数据

### PNG优化
- 使用optipng优化
- 移除元数据

### WebP转换
- 质量设置：80%
- 支持透明度

### 响应式图片
- 生成尺寸：320w, 640w, 768w, 1024w, 1200w
- 同时提供WebP和原格式

## 使用建议

### HTML中使用响应式图片：
```html
<picture>
  <source srcset="image-320w.webp 320w, image-640w.webp 640w, image-768w.webp 768w" type="image/webp">
  <source srcset="image-320w.jpg 320w, image-640w.jpg 640w, image-768w.jpg 768w" type="image/jpeg">
  <img src="image-640w.jpg" alt="描述" loading="lazy">
</picture>
```

### Hugo shortcode使用：
```
{{< responsive-image src="image.jpg" alt="描述" >}}
```

