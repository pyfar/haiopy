#%%



import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd



downsample = 10 # Display every n-th sample
interval = 30  #minimum time between plot updates
channels = [1] #inputchannel to plot -----LIST!!!!
window = 5000 # visible time-slot
samplerate = 48000


c = 2 # Channels

mapping = [c - 1 for c in channels]

q = queue.Queue()




def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::downsample, mapping])
    #q.put(indata.copy())
    

def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines

try:
    

    length = int(window * samplerate / (1000 * downsample))
    plotdata = np.zeros((length, len(channels)))

    fig, ax = plt.subplots()
    lines = ax.plot(plotdata)
    if len(channels) > 1:
        ax.legend(['channel {}'.format(c) for c in channels],
                  loc='lower left', ncol=len(channels))
    ax.axis((0, len(plotdata), -1, 1))
    ax.set_yticks([0])
    ax.yaxis.grid(True)
    ax.tick_params(bottom=False, top=False, labelbottom=False,
                   right=False, left=False, labelleft=False)
    fig.tight_layout(pad=0)

    stream = sd.InputStream(
        device=0, channels=max(channels),
        samplerate=samplerate, callback=audio_callback, blocksize= int(48000 * 50 /1000))
    ani = FuncAnimation(fig, update_plot, interval=interval, blit=True)
    with stream:
        plt.show()
except Exception as e:
    exit(type(e).__name__ + ': ' + str(e))





# %%

# %%
