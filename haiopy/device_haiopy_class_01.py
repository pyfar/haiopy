#%%

import sounddevice as sd
import time


class Device: 

    def __init__(self):
        

        self.input = input
        self.output = output
        
    

         


    def device_use(self):
        return self.input,self.output

    def device_set(self):
        sd.default.device = (self.input,self.output)

    def show_input_settings(self):
        print(sd.query_devices(device=self.input))

    def show_output_settings(self):
        print(sd.query_devices(device=self.output))

    def channels_show(self):
        print('Max Channels for Input Device:')
        print(sd.query_devices(device=self.input)['max_input_channels'])
        print('Max Channels for Output Device:')
        print(sd.query_devices(device=self.output)['max_output_channels'])


    def channel_select(self,ichan,ochan):
        sd.default.channels = ichan, ochan
        



#%%

test_device = Device(0,2)



test_device.device_set()

print(sd.query_devices())

test_device.device_use()

#%%

change = Device(0,1)

change.device_set()

print(sd.default.device)

change.show_input_settings()
#sd.query_devices(0)
time.sleep(1)

print(80*'#')

change.show_output_settings()


time.sleep(1)

print(80*'#')

change.channels_show()
time.sleep(1)

print(sd.default.channels)
time.sleep(1)
print(80*'#')

change.channel_select(1,2)

print(sd.default.channels)




# %%
