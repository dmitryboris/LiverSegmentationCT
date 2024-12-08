from django.forms import FileField, FileInput, Form

class DCMFileUploadForm(Form):
    dcm_file = FileField(
        label="Выберите DCM файл",
        widget=FileInput(attrs={
            'class': 'file-input',
        })
    )
