import argparse
import itertools
import pprint
import progressbar
import random
import requests
import time
import cv2
import numpy as np


class APIError(Exception):
    """An API Error Exception"""

    def __init__(self, status):
        self.status = status

    def __str__(self):
        return "APIError: {}".format(self.status)


class FirenodejsAPI:
    def __init__(self, args):
        self._url = args.url
        self._fake_move = args.fakemove
        if args.tv is not None:
            self.position({'sys':{'tv': args.tv}})
        if args.mv is not None:
            self.position({'sys':{'mv': args.mv}})

        if not self._fake_move:
            response = self.position(request={'sys':''})
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(response)

    
    def camera(self, src='video0'):
        '''Get image from firenodejs REST API.'''
        resp = requests.get(self._url + '/camera/' + src + '/image.jpg')
        if resp.status_code != 200:
            raise APIError('GET /camera/{}/image.jpg {}'.format(src, resp.status_code))
        return cv2.imdecode(np.frombuffer(resp.content, np.uint8), -1)


    def position(self, request={}):
        if request:
            if self._fake_move:
                print(request)
                return
            resp = requests.post(self._url + '/firestep', json=request)
            if resp.status_code != 200:
                raise APIError('POST {}/firestep json={} -> {}'.format(self._url, request, resp.status_code))
            #time.sleep(0.01)
            return resp.json()


#resp = requests.get('http://10.0.0.10:8080/images/default/image.jpg')
#resp = requests.get('http://10.0.0.10:8080/firestep/model')
#if resp.status_code != 200:
#    raise APIError('GET /firestep/model/ {}'.format(resp.status_code))
#print(resp.content)
#for item in resp.json():
#    print('{}: {}'.format(item, resp.json()[item]))


def set_z(api, z):
    api.position({'mov': {'z': z}})


def move_to_xy(api, x, y):
    api.position({'mov': {'x': x, 'y': y, 'lpp': False}})


def move_to_xyz(api, x, y, z):
    api.position({'mov': {'x': x, 'y': y, 'z': z, 'lpp': False}})


def draw_rectangle(api, x0, y0, x1, y1, *, fill=False):
    # outline
    move_to_xy(api, x0, y0)
    set_z(api, -10)
    move_to_xy(api, x0, y1)
    move_to_xy(api, x1, y1)
    move_to_xy(api, x1, y0)
    move_to_xy(api, x0, y0)

    # fill
    if fill:
        dx = 1
        move_to_xy(api, min(x0, x1), y0)
        for x in range(min(x0, x1), max(x0, x1)+dx, dx):
            move_to_xy(api, x, y0)
            move_to_xy(api, x, y1)
    set_z(api, 0)


def draw_chessboard(api, args):
    api.position(request={'hom':''})
    chessboard(api, (0, 0), args.size)


def chessboard(center, size):
    '''Draws a 8x8 chessboard.'''
    a = size / 8
    x0 = int(floor(center[0] - size / 2))
    y0 = int(floor(center[1] - size / 2))
    x1 = int(ceil(center[0] + size / 2))
    y1 = int(ceil(center[1] + size / 2))

    draw_rectangle(x0, y0, x1, y1)
    for x, y in itertools.product(range(-4, 4), range(-4, 4)):
        if (x + y) % 2 != 0:
            continue
        draw_rectangle(x * a, y * a, (x + 1) * a, (y + 1) * a, fill=True)


def draw_bitmap(api, args):
    api.position(request={'hom':''})
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if args.hflip:
        img = img[:, ::-1]
    if args.dots:
        draw_bitmap_dots(api, img, args)
    if args.lines:
        draw_bitmap_lines(api, img, args)


def draw_bitmap_dots(api, img, args):
    height, width = img.shape[:2]
    dots = bitmap_2_dots(img, threshold=args.threshold)
    x0, y0 = tuple(map(float, args.offset.split(':')))
    if args.center:
        x0 = x0 - width / args.dpmm / 2
        y0 = y0 - height / args.dpmm / 2

    # apply scale and offset
    dots = list(((x / args.dpmm + x0, y / args.dpmm + y0) for (x, y) in dots))

    print('Drawing bitmap: {}x{}px, {}x{}mm, offset: {}:{}mm'.format(
        width, height, width / args.dpmm, height /args.dpmm, x0, y0
    ))
    print('# dots:', len(dots))
    time.sleep(2)

    if args.random:
        random.shuffle(dots)

    widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ]
    bar = progressbar.ProgressBar(max_value=len(dots), widgets=widgets).start()
    draw_dots(api, dots, z_up=args.z_up, z_down=args.z_down, cback=bar.update)
    bar.finish()


def bitmap_2_dots(img, *, threshold=128):
    """Convert bitmap image into a generator of point coordinates."""
    assert len(img.shape) == 2
    h, w = img.shape
    for r, c in itertools.product(range(h), range(w)):
        if img[r, c] < threshold:
            yield (c, r)


def draw_dots(api, dots, *, z_up, z_down, cback=None):
    # make a separate move between dots if max(abs(px-x), abs(py-y)) is greater than threshold
    px = 0
    py = 0
    adj_threshold = 4  # [mm]
    for i, (x, y) in enumerate(dots):
        if max(abs(px - x), abs(py - y)) > adj_threshold:
            move_to_xyz(api, x, y, z_up)
        move_to_xyz(api, x, y, z_down)
        move_to_xyz(api, x, y, z_up)
        px = x
        py = y
        if cback is not None:
            cback(i)


def draw_bitmap_lines(api, img, args):
    height, width = img.shape[:2]
    lines = bitmap_2_lines(img, threshold=args.threshold)
    off_x, off_y = tuple(map(float, args.offset.split(':')))
    if args.center:
        off_x = off_x - width / args.dpmm / 2
        off_y = off_y - height / args.dpmm / 2

    print('Drawing bitmap: {}x{}px, {}x{}mm, offset: {}:{}mm'.format(
        width, height, width / args.dpmm, height /args.dpmm, off_x, off_y
    ))
    time.sleep(2)

    # apply scale and offset
    lines = list((
        ((x0 / args.dpmm + off_x, y0 / args.dpmm + off_y), (x1 / args.dpmm + off_x, y1 / args.dpmm + off_y))
        for ((x0, y0), (x1, y1)) in lines
    ))

    if args.random:
        random.shuffle(lines)

    widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.AdaptiveETA(), ') ',
    ]
    bar = progressbar.ProgressBar(max_value=len(lines), widgets=widgets).start()
    draw_lines(api, lines, z_up=args.z_up, z_down=args.z_down, cback=bar.update)
    bar.finish()


def bitmap_2_lines(img, *, threshold=128):
    """Convert a bitmap to a set of horizontal lines."""
    assert len(img.shape) == 2
    h, w = img.shape
    for r in range(h):
        begin = None
        end = None
        for c in range(w):
            if img[r, c] < threshold:
                if begin is None:
                    begin = c
                elif end is not None:
                    yield ((begin, r), (end - 1, r))
                    begin = None
                    end = None
            if img[r, c] >= threshold and begin is not None and end is None:
                end = c
        if begin is not None and end is not None:
            yield ((begin, r), (end - 1, r))


def draw_lines(api, lines, *, z_up, z_down, cback=None):
    set_z(api, z_up)
    for i, ((x0, y0), (x1, y1)) in enumerate(lines):
        api.position({'mov': {'x': x0, 'y': y0}})
        set_z(api, z_down)
        api.position({'mov': {'x': x1, 'y': y1}})
        set_z(api, z_up)
        if cback is not None:
            cback(i)


parser = argparse.ArgumentParser(description='Tool for drawing via firenodejs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers()
parser.add_argument('--url', default='http://10.0.0.10:8080', help='URL of firenodejs')
parser.add_argument('--fakemove', action='store_true', help='Fake movement')
parser.add_argument('--z-up', type=float, default=15, help='Retract tip height')
parser.add_argument('--z-down', type=float, default=-10, help='Draw tip height')
parser.add_argument('--tv', type=float, default=None, help='Set seconds to reach maximum velocity')
parser.add_argument('--mv', type=float, default=None, help='Set maximum velocity (pulses/second)')

parser_bitmap = subparsers.add_parser('bitmap', help='Bitmap drawing')
parser_bitmap.add_argument('--image', help='Path to image', required=True)
parser_bitmap.add_argument('--dots', help='Draw using dots', action='store_true')
parser_bitmap.add_argument('--lines', help='Draw using lines', action='store_true')
parser_bitmap.add_argument('--hflip', help='Flip the image horizontally (around vertical axis)', action='store_true')
parser_bitmap.add_argument('--threshold', help='Threshold for binarizing an image', default=128, type=int)
parser_bitmap.add_argument('--dpmm', type=float, help='Dots per mm', default=1)
parser_bitmap.add_argument('--offset', help='Offset the image [mm]', default='0:0')
parser_bitmap.add_argument('--center', help='Center image around 0:0', action='store_true')
parser_bitmap.add_argument('--random', help='Draw the dots in a random order', action='store_true')
parser_bitmap.set_defaults(func=draw_bitmap)

parser_chess = subparsers.add_parser('chess', help='Chessboard drawing')
parser_chess.add_argument('--size', help='Width of the chessboard', default=80)
parser_chess.set_defaults(func=draw_chessboard)

args = parser.parse_args()
api = FirenodejsAPI(args)
if hasattr(args, 'func'):
    args.func(api, args)



#feed = 'video0'
#print('Press ESC to end')
#while True:
#    image = api.camera(src=feed)
#    cv2.imshow(feed, image)
#    k = cv2.waitKey(1)
#    if k == 27:
#        break
