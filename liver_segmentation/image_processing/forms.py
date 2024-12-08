from django import forms


class DCMFileUploadForm(forms.Form):
    dcm_files = forms.FileField(
        label="Выберите DCM файлы",
        widget=forms.ClearableFileInput(attrs={
            'class': 'file-input',
            'multiple': True
        })
    )

    def clean_dcm_files(self):
        files = self.files.getlist('dcm_files')
        if not files:
            raise forms.ValidationError("Вы должны загрузить хотя бы один файл.")

        return files