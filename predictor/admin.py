
# Register your models here.
from django.contrib import admin
from .models import ProfilAdverse, Argument, Negociation

admin.site.register(ProfilAdverse)
admin.site.register(Argument)
admin.site.register(Negociation)