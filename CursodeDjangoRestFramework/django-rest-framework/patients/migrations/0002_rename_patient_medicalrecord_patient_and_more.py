# Generated by Django 5.1.1 on 2024-10-02 01:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('patients', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='medicalrecord',
            old_name='Patient',
            new_name='patient',
        ),
        migrations.AlterField(
            model_name='medicalrecord',
            name='date',
            field=models.DateField(),
        ),
    ]
