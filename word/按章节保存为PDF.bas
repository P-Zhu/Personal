Attribute VB_Name = "NewMacros"
Sub 按章节保存为PDF()
Dim i%, aCount%
n = 1 'Page_start
t = 0 'Page_end
fn = Split(ActiveDocument.Name, ".")(0)


With ActiveDocument()
    aCount = .Sections.Count
    For i = 1 To aCount

        t = .Sections(i).Range.ComputeStatistics(wdStatisticPages) - IIf(i = .Sections.Count, 0, 1)    '获取该节总页数
        ActiveDocument.ExportAsFixedFormat OutputFileName:=fn & Format(i, "00") & ".pdf", ExportFormat _
        :=wdExportFormatPDF, OpenAfterExport:=True, OptimizeFor:= _
        wdExportOptimizeForOnScreen, Range:=wdExportFromTo, From:=n, To:=n + t - 1, Item _
        :=wdExportDocumentContent, IncludeDocProps:=True, KeepIRM:=True, _
        CreateBookmarks:=wdExportCreateNoBookmarks, DocStructureTags:=True, _
        BitmapMissingFonts:=True, UseISO19005_1:=True
        n = n + t
    Next i
End With

End Sub

