# Generated by Django 5.1 on 2024-08-18 22:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('my_first_app', '0002_car_year'),
    ]

    operations = [
        migrations.AddField(
            model_name='car',
            name='color',
            field=models.TextField(max_length=50, null=True),
        ),
        migrations.AddField(
            model_name='car',
            name='model',
            field=models.TextField(max_length=250, null=True),
        ),
    ]