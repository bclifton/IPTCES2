import sys
from PIL import ImageFont, ImageDraw, Image

def draw(name, country, prediction, suggestion, graph_file):
    size = (1524*2, 750*2)
    margin = 220
    topmargin = 120
    graphsize = (1968, 560)

    regular_fontpath = 'assets/AkzidenzGrotesk/AkzidGroBEReg'
    bold_fontpath = 'assets/AkzidenzGrotesk/AkzidGroBEBol'

    canvas = Image.new('RGBA', size, (0,0,0))
    draw = ImageDraw.Draw(canvas)

    ###############################
    # Dear ______
    ###############################
    reg = ImageFont.truetype(filename=regular_fontpath, size=80)
    bold = ImageFont.truetype(filename=bold_fontpath, size=80)

    sentence = ["Dear ", name + " ", "of ", country, ","]
    bolded = [1, 3]

    x = margin
    y = topmargin
    for i, word in enumerate(sentence):
        if i in bolded:
            f = bold
        else:
            f = reg

        sz = f.getsize(word)
        draw.text((x, y), word, font=f, fill=(255, 255, 255))
        x += sz[0]


    ###############################
    # Prediction label
    ###############################
    y += 110
    draw.text((margin, y), "This week we predict that you will:", font=reg, fill=(255, 255, 255))

    ###############################
    # Prediction text
    ###############################
    y += 110
    pred_fontsize = 100
    pred = ImageFont.truetype(filename=bold_fontpath, size=pred_fontsize)

    fit = False
    while fit is False:
        sz = pred.getsize(prediction)
        if sz[0] <= graphsize[0]:
            fit = True
        else:
            pred_fontsize -= 1
            pred = ImageFont.truetype(filename=bold_fontpath, size=pred_fontsize)

    draw.text((margin, y), prediction, font=pred, fill=(244, 85, 74))


    ###############################
    # Graph
    ###############################
    y += 110
    graph = Image.open(graph_file).resize(graphsize)
    canvas.paste(graph, mask=graph, box=(margin, y))

    ###############################
    # Suggestion Label
    ###############################
    y += 600
    draw.text((margin, y), "To alleviate the Global Crisis, we suggest that you:", font=reg, fill=(255, 255, 255))

    ###############################
    # Suggestion text
    ###############################
    sug_fontsize = 120
    sug = ImageFont.truetype(filename=bold_fontpath, size=sug_fontsize)
    fit = False
    while fit is False:
        sz = sug.getsize(suggestion)
        if sz[0] <= graphsize[0]:
            fit = True
        else:
            sug_fontsize -= 1
            sug = ImageFont.truetype(filename=bold_fontpath, size=sug_fontsize)

    y += 120
    padding = (20, 10)
    draw.rectangle([margin, y, margin+sz[0]+padding[0]*2, y+sz[1]+padding[1]*2], fill=(75, 146, 219))
    draw.text((margin + padding[0], y + padding[1]), suggestion, font=sug, fill=(255, 255, 255))


    ###############################
    # Logo
    ###############################
    logo = Image.open('assets/logo.png').resize((493, 600))
    canvas.paste(logo, mask=logo, box=(2320, 320))

    return canvas


if __name__ == '__main__':
    canvas = draw(name = "Ellen Johnson Sirleaf", country = "Liberia", prediction = "Make a denial", suggestion = "Make a symbolic statement", graph_file = "assets/graph.png")

    #canvas.save(sys.stdout, "GIF")
    canvas.show()
