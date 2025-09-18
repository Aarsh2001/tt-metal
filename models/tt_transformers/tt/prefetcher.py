from typing import List

import ttnn
from models.common.lightweightmodule import LightweightModule


### Helper class to manage subdevices for the Prefetcher
# The class PrefetcherSubDevice provides an interface for creating subdevices
class PrefetcherSubDevice:
    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.num_sub_devices = 0
        self.sub_devices: List[ttnn.SubDevice] = None
        self.sub_devices_id: List[ttnn.SubDeviceId] = None

    def add_sub_device(self, core_range_set: ttnn.CoreRangeSet):
        self.sub_devices.append(ttnn.SubDevice([core_range_set]))
        self.sub_devices_id.append(ttnn.SubDeviceId(len(self.sub_devices_id)))

    def init_sub_device_manager(self):
        self.manager_id = self.mesh_device.create_sub_device_manager(self.sub_devices, 0)
        self.mesh_device.load_sub_device_manager(self.manager_id)
        self.mesh_device.set_sub_device_stall_group(self.sub_devices_id)


class Prefetcher(LightweightModule):
    def __init__(self, mesh_device: ttnn.MeshDevice, num_tensors: int, num_layers: int, mode: str):
        """
        Prefetcher class that prefetches tensors from DRAM to
        """
        ### Device, Global CB, Parameters
        self.global_cb = glv
        self.mesh_device = mesh_device
        self.num_tensors = num_tensors
        self.num_layers = num_layers
        self.enable_performance_mode = False

        ### Prefetcher Subdevices
        self.prefetcher_sub_device = PrefetcherSubDevice(self.mesh_device)

        ### Prefetched Tensors
        self.prefetched_tensors = []
        self.prefetched_tensor_addr = []

    def init(self, mode: str = "decode") -> None:
        """ """

    def create_prefetcher_cores(self):
        pass

    def run(self):
        """ """
        # Create global cb buffer if it was not
        if self.global_cb is None:
            self.global_cb = ttnn.create_global_circular_buffer(
                self.mesh_device,
                self.sender_receiver_mapping,
                self.global_cb_size,
            )
        # Run prefetcher op
        t = ttnn.dram_prefetcher(
            self.num_tensors,
            num_layers=self.num_layers,
            global_cb=self.global_cb,
            enable_performance_mode=self.enable_performance_mode,
        )
        ttnn.deallocate(t)
        return
