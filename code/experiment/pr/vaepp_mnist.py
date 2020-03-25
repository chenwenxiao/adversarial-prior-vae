import code.experiment.compute_pnr as cpr

redraw=False
path = '/mnt/mfs/mlstorage-experiments/cwx17/b8/2c/d445f4f80a9ff9cea7e5'

if redraw:
    cpr.plot_in_one(path,prefix='vaepp',picname='vaepp-mnist')
else:
    cpr.draw(0)