@echo off
chcp 65001
@echo off
title 批量删除非指定类型文件
echo...  

echo 保留 pdf doc* mp4\n\n
echo...
attrib -R *.* /s>nul 2>nul 
attrib +R *.pdf /s>nul 2>nul 
attrib +R *.doc* /s>nul 2>nul 
attrib +R *.wps* /s>nul 2>nul 
attrib +R *.mp4 /s>nul 2>nul 
attrib +R *.bat /s>nul 2>nul 
attrib +R *.txt /s>nul 2>nul 

echo 正在删除当前目录及子目录中所有的非指定类型文件，请稍后...  
echo...
@del .\*.* /f /s /q /a:-r>a.txt

attrib -R *.pdf /s>nul 2>nul 
attrib -R *.doc* /s>nul 2>nul 
attrib -R *.mp4 /s>nul 2>nul 
echo 完成！


pause 
