import math


class Algorithm:

    def __init__(self, receptive_field: int = 15, no_parameters: int = 5000000,
                 no_maps: int = 5, no_channels_input: int = 350,
                 no_channels_last_spatial_block: int = 64, no_channels_minimum_temporal_block: int = 32,
                 no_channels_decrease_temporal: int = 32, no_channels_decrease_spatial: int = 32,
                 no_non_linearities: int = 21):
        self.receptive_field = receptive_field
        self.no_parameters = no_parameters
        self.no_maps = no_maps

        self.no_channels_input = no_channels_input
        self.no_channels_last_spatial_block = no_channels_last_spatial_block
        self.no_channels_minimum_temporal_block = no_channels_minimum_temporal_block
        self.no_channels_decrease_temporal = no_channels_decrease_temporal
        self.no_channels_decrease_spatial = no_channels_decrease_spatial
        self.no_non_linearities = no_non_linearities

        self.no_channels_temporal_blocks = []  # C_T
        self.no_channels_spatial_blocks = []  # C_S
        self.no_layers_temporal_blocks = []  # L
        self.no_parameters_difference = 0

        self.calculate()

    def get_no_channels_temporal_blocks(self):
        return self.no_channels_temporal_blocks

    def get_no_channels_spatial_blocks(self):
        return self.no_channels_spatial_blocks

    def get_no_layers_temporal_blocks(self):
        return self.no_layers_temporal_blocks

    def calculate(self):
        # calculate number of channels for spatial blocks
        no_blocks = int((self.receptive_field - 1) / 2)
        self.no_channels_spatial_blocks = [i * self.no_channels_decrease_spatial + self.no_channels_last_spatial_block
                                           for i in range(no_blocks)][::-1]

        # calculate remaining non linearities
        no_nonlinearities_remaining = self.no_non_linearities - no_blocks

        # calculate non linearities per temporal block - or in other words the growth rate...
        if no_blocks > 0:
            self.no_layers_temporal_blocks = [no_nonlinearities_remaining // no_blocks] * no_blocks
            for i in range(no_nonlinearities_remaining % no_blocks):
                self.no_layers_temporal_blocks[i] += 1
        else:
            # minus one because we use a spatial block with 1 x 1 instead of 3 x 3 before the last 1 x 1 convolution
            self.no_layers_temporal_blocks = [no_nonlinearities_remaining - 1]

        # to keep track of optimal network setting
        previous_no_params_remaining = math.inf
        previous_temporal_blocks = None

        for ch_init in range(1, 2000, 1):  # 1000 should be fine as maximum
            # calculate remaining parameters considering last 1x1 convolution
            no_params_remaining = self.no_parameters - (1 * 1 * self.no_channels_last_spatial_block * self.no_maps +
                                                        self.no_maps)

            temporal_blocks = []

            if self.receptive_field == 1:
                params, out_ch = self.no_parameters_temporal_block(self.no_channels_input, ch_init,
                                                                   self.no_layers_temporal_blocks[0])
                no_params_remaining -= params
                # instead of a spatial block, we use a simple 1 x 1 convolution to keep the receptive field at 1
                no_params_remaining -= (1 * 1 * out_ch * self.no_channels_last_spatial_block +
                                        self.no_channels_last_spatial_block)
                temporal_blocks.append(ch_init)
            else:
                out_ch = self.no_channels_input
                for i in range(no_blocks):  # as many temporal blocks as spatial blocks
                    params, out_ch = self.no_parameters_temporal_block(out_ch, ch_init,
                                                                       self.no_layers_temporal_blocks[i])
                    no_params_remaining -= params
                    params = self.no_parameters_spatial_block(out_ch, self.no_channels_spatial_blocks[i])
                    no_params_remaining -= params

                    # init for next block
                    out_ch = self.no_channels_spatial_blocks[i]
                    temporal_blocks.append(ch_init)
                    # ch_init = ch_init // 2  # would be another possibility
                    ch_init -= self.no_channels_decrease_temporal
                    if ch_init < self.no_channels_minimum_temporal_block:
                        ch_init = self.no_channels_minimum_temporal_block

            if previous_no_params_remaining > no_params_remaining > 0:
                previous_no_params_remaining = no_params_remaining
                previous_temporal_blocks = temporal_blocks
                ch_init += 1  # increment for next parameter set
            elif previous_no_params_remaining < abs(no_params_remaining):
                # previous setting was better
                self.no_channels_temporal_blocks = previous_temporal_blocks
                self.no_parameters_difference = previous_no_params_remaining
                break
            else:
                self.no_channels_temporal_blocks = temporal_blocks
                self.no_parameters_difference = no_params_remaining
                break

    @staticmethod
    def no_parameters_spatial_block(ch_in: int, ch_out: int):
        return 3 * 3 * ch_in * ch_out + ch_out

    @staticmethod
    def no_parameters_temporal_block(ch_in: int, ch_out: int, no_layers: int = 4):
        return sum([1 * 1 * (ch_in + idx * ch_out) * ch_out + ch_out for idx in range(no_layers)]), \
               (ch_in + no_layers * ch_out)
