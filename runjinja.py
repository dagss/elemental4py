import sys
from jinja2 import Template

# Import template_utils.py, it is injected as "utils" into the template
# namespace
import template_utils

input_filename = sys.argv[1]
f = file(input_filename)
try:
    template = Template(f.read())
finally:
    f.close()

result = template.render(utils=template_utils, len=len)

try:
    output_filename = sys.argv[2]
except IndexError:
    sys.stdout.write(result)
else:
    f = file(output_filename, 'w')
    try:
        f.write(result)
    finally:
        f.close()
