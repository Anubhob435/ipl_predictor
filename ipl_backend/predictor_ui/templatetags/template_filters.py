from django import template
register = template.Library()

@register.filter(name='sub')
def sub(value, arg):
    """Subtracts the arg from the value."""
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        try:
            return int(value) - int(arg)
        except (ValueError, TypeError):
            return 0

@register.filter(name='add_class')
def add_class(value, arg):
    """
    Add a CSS class to form widget
    Usage: {{ form.field.as_widget|add_class:"form-control" }}
    """
    return value.replace('<input ', f'<input class="{arg}" ').\
           replace('<select ', f'<select class="{arg}" ').\
           replace('<textarea ', f'<textarea class="{arg}" ')

@register.filter
def subtract(value, arg):
    """Subtract the arg from the value."""
    try:
        return int(value) - int(arg)
    except (ValueError, TypeError):
        try:
            return float(value) - float(arg)
        except (ValueError, TypeError):
            return value
