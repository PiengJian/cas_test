class Block(nn.Layer):
    """AFNO network block.

    Args:
        dim (int): The input tensor dimension.
        mlp_ratio (float, optional): The ratio used in MLP. Defaults to 4.0.
        drop (float, optional): The drop ratio used in MLP. Defaults to 0.0.
        drop_path (float, optional): The drop ratio used in DropPath. Defaults to 0.0.
        activation (str, optional): Name of activation function. Defaults to "gelu".
        norm_layer (nn.Layer, optional): Class of norm layer. Defaults to nn.LayerNorm.
        double_skip (bool, optional): Whether use double skip. Defaults to True.
        num_blocks (int, optional): The number of blocks. Defaults to 8.
        sparsity_threshold (float, optional): The value of threshold for softshrink. Defaults to 0.01.
        hard_thresholding_fraction (float, optional): The value of threshold for keep mode. Defaults to 1.0.
    """

    def __init__(
        self,
        dim: int,
        input_resolution,
        num_heads,
        window_size=8,
        shift_size=0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        attn_channel_ratio=0.5,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        activation: str = "gelu",
        norm_layer: nn.Layer = nn.LayerNorm,
        double_skip: bool = True,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ):
        super().__init__()
        self.dim = dim-192
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn_channle_ratio = attn_channel_ratio
        self.attn_dim = int(self.dim * attn_channel_ratio)
        self.cnn_dim = 192

        self.norm1 = norm_layer(dim)

        if self.dim - self.attn_dim > 0:
            self.filter = AFNO2D(
                self.dim - self.attn_dim,
                num_blocks,
                sparsity_threshold,
                hard_thresholding_fraction,
            )
        if self.attn_dim > 0:
            self.attn = WindowAttention(
                self.attn_dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )

            if self.shift_size > 0:
                # calculate attention mask for SW-MSA
                H, W = self.input_resolution
                Hp = int(np.ceil(H / self.window_size)) * self.window_size
                Wp = int(np.ceil(W / self.window_size)) * self.window_size
                img_mask = paddle.zeros([1, Hp, Wp, 1], dtype="float32")  # 1 Hp Wp 1
                h_slices = (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None),
                )
                w_slices = (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None),
                )
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        try:
                            img_mask[:, h, w, :] = cnt
                        except:
                            pass

                        cnt += 1

                mask_windows = window_partition(
                    img_mask, self.window_size
                )  # nW, window_size, window_size, 1
                mask_windows = mask_windows.reshape(
                    [-1, self.window_size * self.window_size]
                )
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                huns = -100.0 * paddle.ones_like(attn_mask)
                attn_mask = huns * (attn_mask != 0).astype("float32")
            else:
                attn_mask = None

            self.register_buffer("attn_mask", attn_mask)
        self.cnn = paddle.nn.Conv2D(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            drop=drop,
        )
        self.double_skip = double_skip

    def attn_forward(self, x):
        B, H, W, C = x.shape

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, [0, pad_l, 0, pad_b, 0, pad_r, 0, pad_t])
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(
                x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(
            [-1, self.window_size * self.window_size, C]
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp, C
        )  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2)
            )
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]
        return x

    def afno_forward(self, x):
        x = self.filter(x)
        return x
    
    def cnn_forward(self,x):
        x = x.transpose([0, 3, 1, 2])
        x = self.cnn(x)
        x = x.transpose([0, 2, 3, 1])

        return x

    def forward(self, x):
        B, H, W, C = x.shape
        residual = x
        x = self.norm1(x)

        # if self.attn_dim == 0:
        #     x = self.afno_forward(x)
        # elif self.attn_dim == self.dim:
        #     x = self.attn_forward(x)
        # else:  # self.attn_dim > 0 and self.attn_dim < self.dim
        x_attn = x[:, :, :, : self.attn_dim]
        x_afno = x[:, :, :, self.attn_dim : -self.cnn_dim ]
        x_cnn = x[:, :, :, -self.cnn_dim :]

        x_attn = self.attn_forward(x_attn)
        x_afno = self.afno_forward(x_afno)
        x_cnn = self.cnn_forward(x_cnn)


        x = paddle.concat([x_attn, x_afno, x_cnn], axis=-1)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x