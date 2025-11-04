import proplot as pplt
import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 20
predict = np.load('./output/rb/pretrain/x_1.npy')
real = np.load('./output/rb/pretrain/y_1.npy')

predict_t2 = predict[0, -1]
real_t2 = real[0, -1]

fig, axs = pplt.subplots(
    ncols=2, nrows=1, refwidth=6, refheight=4, sharey=False, sharex=True
)
s = axs[0].contourf(real_t2, cmap="jet", extend="both", levels=np.linspace(-2, 2, 100))
s = axs[1].contourf(predict_t2, cmap="jet", extend="both", levels=np.linspace(-2, 2, 100))
axs.format(fontsize=FONTSIZE, title="T2")
fig.colorbar(
    s, loc="r", ticklabelsize=FONTSIZE, labelsize=FONTSIZE, shrink=1, extendsize=0.5, locator=0.5
)

plt.savefig('./T2预报图.png', bbox_inches='tight')


predict_t2 = predict[0, -3]
real_t2 = real[0, -3]

fig, axs = pplt.subplots(
    ncols=2, nrows=1, refwidth=6, refheight=4, sharey=False, sharex=True
)
s = axs[0].contourf(real_t2, cmap="jet", extend="both", levels=np.linspace(-2, 2, 100))
s = axs[1].contourf(predict_t2, cmap="jet", extend="both", levels=np.linspace(-2, 2, 100))
axs.format(fontsize=FONTSIZE, title="U10")
fig.colorbar(
    s, loc="r", ticklabelsize=FONTSIZE, labelsize=FONTSIZE, shrink=1, extendsize=0.5, locator=0.5
)

plt.savefig('./U10预报图.png', bbox_inches='tight')
