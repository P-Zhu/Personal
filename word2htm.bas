Attribute VB_Name = "NewMacros"
Public Function GetOutFolderName(ByVal DialogType As MsoFileDialogType) As String
    '打开目录
    
    With Application.FileDialog(DialogType)
        .AllowMultiSelect = False
        .Title = "选择HTML文件保存位置"
        .ButtonName = "保存到"
        If .Show = True Then
            GetOutFolderName = .SelectedItems(1)
        End If
    End With
End Function
Public Function SelectFile()
    '选择多个文件
    
    Dim i   As Integer
    With Application.FileDialog(msoFileDialogFilePicker)
        .AllowMultiSelect = True
        .Title = "选择文件"
        '单选择
        .Filters.Clear
        '清除文件过滤器
        .Filters.Add "选择要转换的的文件", "*.doc;*.docx;*.docm;*.pdf"
        .Filters.Add "All Files", "*.*"
        '设置两个文件过滤器
        If .Show = -1 Then
            'FileDialog 对象的 Show 方法显示对话框，并且返回 -1（如果您按 OK）和 0（如果您按 Cancel）
            length = .SelectedItems.count
            ReDim Selcted(1 To length)
            For i = 1 To length
                Selcted(i) = .SelectedItems(i)
            Next
            SelectFile = Selcted
        Else
            ReDim Selcted(1)
            SelectFile = Selcted
        End If
        
    End With
End Function
Sub PDF2Html()
    
    On Error Resume Next
    
    Load FormTips   '载入提示框
    
    Dim OutPath As String, _
        FName As String, _
        InFile, _
        length As Integer, _
        Selected()
        
    Selected = SelectFile()
    If IsEmpty(Selected(1)) Then
        Exit Sub
    End If
     
    OutPath = GetOutFolderName(msoFileDialogFolderPicker)
    If OutPath = "" Then
        Exit Sub
    End If
    
    For Each InFile In Selected
      length = length + 1
    Next
    ChangeFileOpenDirectory (OutPath)
    
   ' Set FSO = CreateObject("Scripting.FileSystemObject")    '创建计算机文件系统以向其访问
    'Set FDR = FSO.GetFolder(InPath)                         '指定其中访问的文件夹对象
   ' Set FList = FDR.Files                                   '定义该文件夹中的所有文件集合
    'FormTips.Caption = "正在处理..."
    'FormTips.OutInfo = "正在打开文件……"
    'FormTips.Show 0
    i = 1
    
    Application.StatusBar = "正在打开文件->"
    
    For Each InFile In Selected                                      '在指定文件夹中循环
    Application.StatusBar = "正在打开文件->" & i & "/" & length & "   " & InFile
        'FormTips.OutInfo = "打开文件" + vbCrLf + InFile
        
        Documents.Open FileName:=InFile, _
            ConfirmConversions:=False, ReadOnly:=True, AddToRecentFiles:=False, _
            PasswordDocument:="", PasswordTemplate:="", Revert:=False, _
            WritePasswordDocument:="", WritePasswordTemplate:="", Format:= _
            wdOpenFormatAuto, XMLTransform:=""
            
        FileName = ActiveDocument.Name
       ' FormTips.Caption = "正在处理..." & i & "/" & length
       ' FormTips.OutInfo = "正在转换..."
        'DoEvents
        Application.StatusBar = "正在转换..." & i & "/" & length & "   " & ActiveDocument.Name
        i = i + 1
       ' Application.Top = "正在处理..." & i & "/"
        
        ActiveProtectedViewWindow.Edit

        ActiveDocument.SaveAs2 FileName:=FileName & ".html", _
            FileFormat:=wdFormatHTML, LockComments:=False, Password:="", AddToRecentFiles:=True, WritePassword _
            :="", ReadOnlyRecommended:=False, EmbedTrueTypeFonts:=False, _
            SaveNativePictureFormat:=False, SaveFormsData:=False, SaveAsAOCELetter:= _
            False, CompatibilityMode:=0
        ActiveDocument.Close
    Next
    
   ' Unload FormTips
    MsgBox "数据处理完毕！程序退出！", 64, "系统提示"
            DoEvents

    SendKeys "%{F4}", True
End Sub
Sub 宏1()
Attribute 宏1.VB_ProcData.VB_Invoke_Func = "Normal.NewMacros.宏1"
'
' 宏1 宏
'
'
    ActiveProtectedViewWindow.Edit
    ChangeFileOpenDirectory "D:\Cache\Git\Personal\Paper\"
    ActiveDocument.SaveAs2 FileName:= _
        "DeepPath A Reinforcement Learning Method for Knowledge Graph Reasoning.docx" _
        , FileFormat:=wdFormatXMLDocument, LockComments:=False, Password:="", _
        AddToRecentFiles:=True, WritePassword:="", ReadOnlyRecommended:=False, _
        EmbedTrueTypeFonts:=False, SaveNativePictureFormat:=False, SaveFormsData _
        :=False, SaveAsAOCELetter:=False, CompatibilityMode:=15
End Sub
