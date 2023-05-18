#Hires. fix Plus
Add more optional parameters and ToMe to Stable Diffusion WebUI's Hires. fix  

##About
This extension can add more available parameters and Token Merging support to SD WebUI's "Hires. fix" by hijacking txt2img output and running a custom img2img process  

##Usage
If you want to use Hires.fix+, please check the original `Hires.fix` checkbox first, then check `Enable Hires.fix+` under the `Seed` textbox.  
After that you can set the `CFG Scale`, `Sub-Prompt`, `Hires Sampler`...etc parameters.  
If `Use Smart-Steps` is enabled, the extension will use the $\log_{s}{20}\cdot ds$ formula to calculate the most cost-effective number of iterations.  
And if `ToMe for Hires. fix` is enabled, the extension will use `Token Merging`(tomesd) during the Hires. fix process to increase the iteration speed.

##Screenshot
<img src="./images/ui.jpg"/>

##Install
1. Go to SD WebUI's extension tab
2. Click `Install from URL` subtab
3. Paste `https://github.com/lihaoyun6/sd-webui-Hires-fix-Plus` into the URL textbox
4. Click Install and wait for it to complete
5. Once completed, the WebUI needs to be reloaded

##Credits
- Inspired by [stable-diffusion-webui-hires-fix-progressive](https://github.com/Kahsolt/stable-diffusion-webui-hires-fix-progressive) @Kahsolt  
- [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) @AUTOMATIC1111  
- [Token Merging for Stable Diffusion](https://github.com/dbolya/tomesd) @Bolya, Daniel and Hoffman, Judy  
