import base64, json
from io import BytesIO
from urllib import request
from PIL import Image

URL = 'http://127.0.0.1:8080/api/triage'

# Create dummy 200x200 PNG
img = Image.new('RGB', (200, 200), (220, 220, 220))
b = BytesIO(); img.save(b, format='PNG')
img_b64 = base64.b64encode(b.getvalue()).decode('ascii')

payload = {
    'image_base64': img_b64,
    'roi_polygon': [[0.4,0.4],[0.6,0.4],[0.6,0.6],[0.4,0.6]],
    'site': 'nose_tip',
    'age': 54,
    'sex': 'F',
    'risks': {
        'smoker': True,
        'uv_high': True,
        'immunosuppressed': False,
        'radiated': False,
        'anticoagulants': False,
        'diabetes': False,
        'ill_defined_borders': False,
        'recurrent_tumor': False
    },
    'offline': True
}

data = json.dumps(payload).encode('utf-8')
req = request.Request(URL, data=data, headers={'Content-Type': 'application/json'})
with request.urlopen(req, timeout=10) as resp:
    body = resp.read().decode('utf-8')
    print('STATUS', resp.status)
    j = json.loads(body)
    print('FIELDS', sorted(list(j.keys())))
    # quick checks
    assert 'diagnostics' in j and 'overlay' in j and 'oncology' in j
    print('TOP3', j['diagnostics']['top3'])
    print('SAFETY', j['safety'])
