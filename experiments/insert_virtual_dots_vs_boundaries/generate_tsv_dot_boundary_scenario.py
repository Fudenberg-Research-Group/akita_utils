#!/usr/bin/env python

# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

###################################################

import pandas as pd
import numpy as np

from akita_utils.tsv_gen_utils import add_orientation, add_const_flank_and_diff_spacer, add_background

# read tsv with all filetered mouse ctcfs
all_ctcf_path = "./../../data_filtered_ctcf_table/filtered_base_mouse_ctcf.tsv"
df = pd.read_csv(all_ctcf_path, sep="\t")

# we want to perform insertion experiment using all 10 backgrounds for a model
nr_sites = len(df)
nr_backgrounds = 10

# adding bg_index and experiment ID
df = add_background(df, [bg_index for bg_index in range(nr_backgrounds)] )

exp_id = [i for i in range(nr_sites * nr_backgrounds)]
df["exp_id"] = exp_id

# preparing tsv for the boundary scenario

# boundary scenaro parameters
orientation = ["<>"]
flank = 20
spacing_list = [70]

boundary_df = add_orientation(df, orientation_strings=orientation, all_permutations=False)
boundary_df = add_const_flank_and_diff_spacer(boundary_df, flank, spacing_list)
boundary_df.to_csv("./input_data/filtered_base_mouse_ctcf_boundary.tsv", sep = "\t", index=False)

# preparing tsv for the dot scenario

# dot scenaro parameters
orientation = ["><"]
flank = 20
spacing_list = [199980]

dot_df = add_orientation(df, orientation_strings=orientation, all_permutations=False)
dot_df = add_const_flank_and_diff_spacer(dot_df, flank, spacing_list)
dot_df.to_csv("./input_data/filtered_base_mouse_ctcf_dot.tsv", sep = "\t", index=False)

