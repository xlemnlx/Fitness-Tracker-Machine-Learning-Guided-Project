# PART - 3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ----------------------------------------------------------------------------------------------------
# Loop over all combinations and export for both sensors
# This covers all the individual tasks that was made from the 
# now OLD python file.
# ----------------------------------------------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

mpl.style.use("seaborn-v0_8-deep") # Name of the style to be use.
mpl.rcParams["figure.dpi"] = 100

label_list = df["label"].unique()
participant_list = df["participant"].unique()

for per_label in label_list:
    for per_participant in participant_list:
        combined_plot_df = (
            df.query(f"label == '{per_label}'")
            .query(f"participant == '{per_participant}'")
            .reset_index()
                    )
        
        if len(combined_plot_df) > 0:
            # If the code is too long, you can encapsulate it 
            # using () and start on diving the line to 
            # multi-line. Like the code below:
            fig, ax = (
                plt.subplots(
                    nrows=2, 
                    sharex=True, 
                    figsize=(20, 10)
                    )
                )
            combined_plot_df[["acce_x", "acce_y", "acce_z"]].plot(ax=ax[0]) 
            combined_plot_df[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax[1])

            ax[0].legend(
                loc="upper center", 
                bbox_to_anchor=(0.5, 1.15), 
                ncol=3, fancybox=True, 
                shadow=True
                )
            ax[1].legend(
                loc="upper center", 
                bbox_to_anchor=(0.5, 1.15), 
                ncol=3, fancybox=True, 
                shadow=True
                )
            ax[1].set_xlabel("samples")
            
            plt.savefig(f"../../reports/figures/{per_label.title()} ({per_participant}).png")
            plt.show()