from django.shortcuts import render
import joblib

# Create your views here.
def home(request):
    if request.method == 'POST':
        model = joblib.load('./my.joblib')
        radius_mean = float(request.POST['radius_mean'])
        texture_mean = float(request.POST['texture_mean'])
        perimeter_mean = float(request.POST['perimeter_mean'])
        area_mean = float(request.POST['area_mean'])
        smoothness_mean = float(request.POST['smoothness_mean'])
        compactness_mean = float(request.POST['compactness_mean'])
        concavity_mean = float(request.POST['concavity_mean'])
        concave_points_mean = float(request.POST['concave_points_mean'])
        symmetry_mean = float(request.POST['symmetry_mean'])
        fractal_dimension_mean = float(request.POST['fractal_dimension_mean'])
        

        md = model.predict([[radius_mean, texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean]])
        return render(request, 'index.html', {'prediction': md[0]})
    return render(request, "index.html")
