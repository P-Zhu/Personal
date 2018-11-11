Attribute VB_Name = "NewMacros"
Public Function GetOutFolderName(ByVal DialogType As MsoFileDialogType) As String
    '��Ŀ¼
    
    With Application.FileDialog(DialogType)
        .AllowMultiSelect = False
        .Title = "ѡ��HTML�ļ�����λ��"
        .ButtonName = "���浽"
        If .Show = True Then
            GetOutFolderName = .SelectedItems(1)
        End If
    End With
End Function
Public Function SelectFile()
    'ѡ�����ļ�
    
    Dim i   As Integer
    With Application.FileDialog(msoFileDialogFilePicker)
        .AllowMultiSelect = True
        .Title = "ѡ���ļ�"
        '��ѡ��
        .Filters.Clear
        '����ļ�������
        .Filters.Add "ѡ��Ҫת���ĵ��ļ�", "*.doc;*.docx;*.docm;*.pdf"
        .Filters.Add "All Files", "*.*"
        '���������ļ�������
        If .Show = -1 Then
            'FileDialog ����� Show ������ʾ�Ի��򣬲��ҷ��� -1��������� OK���� 0��������� Cancel��
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
    
    Load FormTips   '������ʾ��
    
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
    
   ' Set FSO = CreateObject("Scripting.FileSystemObject")    '����������ļ�ϵͳ���������
    'Set FDR = FSO.GetFolder(InPath)                         'ָ�����з��ʵ��ļ��ж���
   ' Set FList = FDR.Files                                   '������ļ����е������ļ�����
    'FormTips.Caption = "���ڴ���..."
    'FormTips.OutInfo = "���ڴ��ļ�����"
    'FormTips.Show 0
    i = 1
    
    Application.StatusBar = "���ڴ��ļ�->"
    
    For Each InFile In Selected                                      '��ָ���ļ�����ѭ��
    Application.StatusBar = "���ڴ��ļ�->" & i & "/" & length & "   " & InFile
        'FormTips.OutInfo = "���ļ�" + vbCrLf + InFile
        
        Documents.Open FileName:=InFile, _
            ConfirmConversions:=False, ReadOnly:=True, AddToRecentFiles:=False, _
            PasswordDocument:="", PasswordTemplate:="", Revert:=False, _
            WritePasswordDocument:="", WritePasswordTemplate:="", Format:= _
            wdOpenFormatAuto, XMLTransform:=""
            
        FileName = ActiveDocument.Name
       ' FormTips.Caption = "���ڴ���..." & i & "/" & length
       ' FormTips.OutInfo = "����ת��..."
        'DoEvents
        Application.StatusBar = "����ת��..." & i & "/" & length & "   " & ActiveDocument.Name
        i = i + 1
       ' Application.Top = "���ڴ���..." & i & "/"
        
        ActiveProtectedViewWindow.Edit

        ActiveDocument.SaveAs2 FileName:=FileName & ".html", _
            FileFormat:=wdFormatHTML, LockComments:=False, Password:="", AddToRecentFiles:=True, WritePassword _
            :="", ReadOnlyRecommended:=False, EmbedTrueTypeFonts:=False, _
            SaveNativePictureFormat:=False, SaveFormsData:=False, SaveAsAOCELetter:= _
            False, CompatibilityMode:=0
        ActiveDocument.Close
    Next
    
   ' Unload FormTips
    MsgBox "���ݴ�����ϣ������˳���", 64, "ϵͳ��ʾ"
            DoEvents

    SendKeys "%{F4}", True
End Sub
Sub ��1()
Attribute ��1.VB_ProcData.VB_Invoke_Func = "Normal.NewMacros.��1"
'
' ��1 ��
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
