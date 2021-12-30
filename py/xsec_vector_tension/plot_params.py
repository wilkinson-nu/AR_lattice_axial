import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fig_width = 6.75 # in inches, 2x as wide as APS column
gr        = 1.618034333 # golden ratio
plot_axes_default  = (0.12, 0.15, 0.87, 0.83)
figure_size_default = (fig_width, fig_width / gr)

fs_p    = {"fontsize": 24} # font size of text, label, ticks
gA_fs_p = {"fontsize": 22} # font size of ga label
ls_p    = {"labelsize": 14}
# unfilled circle
errorp = {"markersize": 5, "linestyle": "none", "capsize": 3, "elinewidth": 1, "mfc": "none" }
# solid circle
errorb = {"markersize": 5, "linestyle": "none", "capsize": 3, "elinewidth": 1}

default_symbols = ['o','s','^','v','*','h','p','P']

## "Rainbow has infinite colors ok" -Jason
grey      = "#808080"
red       = "#FF6F6F"
peach     = "#FF9E6F"
orange    = "#FFBC6F"
sunkist   = "#FFDF6F"
yellow    = "#FFEE6F"
lime      = "#CBF169"
green     = "#5CD25C"
turquoise = "#4AAB89"
blue      = "#508EAD"
grape     = "#635BB1"
violet    = "#7C5AB8"
fuschia   = "#C3559F"

default_base_colors = [
  fuschia,    violet,  grape,  blue,
  turquoise,  green,   lime,   yellow,
  sunkist,    orange,  peach,  red,
]

def color_fader( c1, c2, mix=0):
    ''' fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        taken from Markus Dutschke answer at stack overflow
        https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
    '''
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

## mix_slider should be within range [0,1]
## will do linear interpolation between colors in base_colors
def single_color_fader( mix_slider, base_colors=default_base_colors):
    if mix_slider < 0.:
        return base_colors[0]
    elif mix_slider >= 1.:
        return base_colors[-1]
    N = len(base_colors)
    mix = (mix_slider % (1./(N-1.)))*(N-1.)
    iwheel = int(mix_slider*(N-1.))
    return color_fader(base_colors[iwheel], base_colors[iwheel+1], mix)

## build a default color map with N equally-spaced colors
def build_color_map( N, base_colors=default_base_colors):
  return [ single_color_fader( i/(N-1), base_colors=base_colors) for i in range(N) ]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    Rmax = 1.1
    Rmin = 0.1
    Nmax = 200
    xmax = 100
    for n in range(Nmax):
        r = Rmax -(Rmax-Rmin)*n/(Nmax-1)
        color = single_color_fader(n/(Nmax-1))
        x_range = np.arange(-r,r+1e-6,2*r/(xmax-1))
        y_range = np.sqrt(np.abs(r*r-x_range*x_range))
        plt.plot(x_range,y_range,color=color,linewidth=1.5)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.show()

