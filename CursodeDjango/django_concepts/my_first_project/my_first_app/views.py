from django.shortcuts import render
from my_first_app.models import Car, Profile
from django.http import HttpResponse
from django.views.generic.base import TemplateView


# Create your views here.
def my_view(request):
    car_list = Car.objects.all()
    context = {"car_list": car_list}
    return render(request, "my_first_app/car_list.html", context)


class CarListView(TemplateView):
    template_name = "my_first_app/car_list.html"

    def get_context_data(self):
        car_list = Car.objects.all()
        return {"car_list": car_list}


def my_test_view(request, *args, **kwargs):
    print(args)
    print(kwargs)
    return HttpResponse("")


def author_profile_show(request, id):
    profile = Profile.objects.get(author_id=id)
    context = {"profile": profile}
    return render(request, "my_first_app/author_profile.html", context)
