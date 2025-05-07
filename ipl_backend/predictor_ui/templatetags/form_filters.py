from django import template
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter(name='tailwind_class')
def tailwind_class(form_field):
    """
    Replaces Bootstrap form-control class with Tailwind CSS classes
    """
    html = str(form_field)
    
    # Replace form-control with Tailwind classes for input fields
    html = html.replace('class="form-control"', 
                       'class="w-full px-4 py-2 text-gray-700 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"')
    
    # Replace form-control with Tailwind classes for select fields
    html = html.replace('class="form-control"', 
                       'class="w-full px-4 py-2 text-gray-700 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"')
    
    return mark_safe(html)

@register.filter(name='add_class')
def add_class(value, arg):
    """
    Add a CSS class to form widget
    Usage: {{ form.field.as_widget|add_class:"form-control" }}
    """
    return value.replace('<input ', f'<input class="{arg}" ').\
           replace('<select ', f'<select class="{arg}" ').\
           replace('<textarea ', f'<textarea class="{arg}" ')