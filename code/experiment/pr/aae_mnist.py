import code.experiment.compute_pnr as cpr

redraw=False
path = ''

if redraw:
    cpr.plot_in_one(path,prefix='aae',picname='aae-mnist')
else:
    cpr.draw(2)