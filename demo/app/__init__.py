from skimage.io import imread

import joblib
import urllib
import cStringIO
import base64

from decaf.scripts.imagenet import DecafNet

from flask import Flask, render_template, request, Markup

app = Flask(__name__)

print('Loading pretrained imagenet model ... ')
net = DecafNet('app/imagenet_pretrained/imagenet.decafnet.epoch90',
               'app/imagenet_pretrained/imagenet.decafnet.meta')
print('[Done]')

print('Loading pretrained SVR model ... ')
clf = joblib.load('app/svr_model.pkl')
print('[Done]')


@app.route('/image_specificity')
def index():
    return render_template('index.html')


@app.route('/image_specificity/predict', methods=['GET', 'POST'])
def predict():
    fileitem, url = None, None
    if request.method == 'GET':
        url = request.args.get('url', None)
        f = cStringIO.StringIO(urllib.urlopen(url).read())
    elif request.method == 'POST':
        fileitem = request.files.get('filename', None)
        f = request.files['filename'].stream

    img = imread(f, False, 'pil')
    scores = net.classify(img, center_only=True)
    fc6 = net.feature('fc6_cudanet_out')
    y_pred = clf.predict(fc6)[0]

    if url is not None:
        img_html = '<img src="%s" height=250></img><br/>' % url
    if fileitem is not None:
        import Image
        output = cStringIO.StringIO()
        I = Image.fromarray(img)
        I.save(output, 'png')
        img = base64.b64encode(output.getvalue()).decode('ascii')
        img_html = ("""<img src="data:image/png;base64,%s"
                       height="250"></img>""" % img)

    return render_template('results.html', img_html=Markup(img_html),
                           y_pred='%0.2f' % y_pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
