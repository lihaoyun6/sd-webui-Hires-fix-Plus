# Hires. fix Plus
本插件通过劫持文生图的采样方法函数来允许用户为高清修复功能设置更多参数, 以及"滚动生成"支持  
(Hires prompts/Hires sampler和Hires ToMe已被SD WebUI主线支持, 故这些功能已从此插件中移除)  

## 截图
<img src="./images/ui.jpg"/>  

## 使用说明
下列是一些名词简介:  

- **Steps optimization** (优化迭代次数)

	> Hires. fix+将会使用公式: $\log_{s}{10}\cdot ds$ 来在保证出图质量的情况下尽可能减少迭代次数, 提升出图速度  
	> 注: 此选项默认启用, 但如果`Hires steps`被设置了数值, 则迭代次数优化将被自动暂时禁用  
	
- **Hires CFG** (高分辨率 CFG Scale)
	> 允许用户为高清修复设置与文生图不同的`CFG Scale`值, 有缓解画风油腻, 画面琐碎, 人脸变形等情况的作用  

- **Rolling factor**
	> 如果值不为 1 则 HF+ 会进行多轮高清迭代. 每次将图像放大此倍数, 直到达到设定的目标放大倍数或分辨率  
	> 注: 可以生成比普通 Hires. fix 更多的画面细节, 还可以提升潜空间采样器的输出质量  
	
- **Settings**
	> 此插件会在设置页面中添加一个名为`Hires. fix+`的选项组, 在这里可修改`Smart-Steps`的最低优化值  
	
## 安装
1. 前往 SD WebUI 的 `扩展` 标签页
2. 点击 `从网址安装` 子标签
3. 将 `https://github.com/lihaoyun6/sd-webui-Hires-fix-Plus` 粘贴进网址输入框
4. 点击 `安装` 并等待完成
5. 提示安装成功后重载 WebUI 即可启用

## 鸣谢
- [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) @AUTOMATIC1111  
