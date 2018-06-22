@echo off  
chcp 65001
@echo off

title 创建文件夹  
echo.  
echo -----------------------------------  
echo 正在创建文件夹，请稍后...  
echo -----------------------------------  

set /a sum=0  
for /f "delims=[" %%i in (folder.txt) do (  
md "%%i"  
set /a sum+=1  
)  
echo ------------------------------------  
echo 共成功创建%cd%目录及其子目录下%sum%个空文件夹！  
echo.  

title 创建文件
echo.  
echo -----------------------------------  
echo 正在创建文件，请稍后...  
echo -----------------------------------  

set /a sum=0  
for /f "delims=[" %%i in (files.txt) do (  
echo %%i
echo.
type nul> "%%i"  
set /a sum+=1  
)  
echo ------------------------------------  
echo 共成功创建%cd%目录及其子目录下%sum%个空文件！  
echo.  

pause 
