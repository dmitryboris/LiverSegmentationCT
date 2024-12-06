from django.http import HttpResponse
from django.shortcuts import render

def home(request):
    return HttpResponse("<h1>Тута будем брать картинку и возвращать готовую</h1>")
