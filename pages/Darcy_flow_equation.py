import sys

import streamlit as st

sys.path.append("..")
from utilities import (
    compute_relative_error,
    plot_data,
    plot_error,
    plot_histogram,
    plot_swarmplot,
    read_data,
)


def darcy_page():
    st.title("Darcy flow equation benchmark", anchor=False)

    st.markdown("This benchmark represents a second-order non-linear PDE, given by:")
    st.latex(
        r"""\begin{cases}
                  -\nabla\cdot (a \nabla u) = f  \quad & in \ D\times (0, T) \\
                  u = 0                                & on\ \partial D
              \end{cases}"""
    )
    st.markdown(
        """
       The dataset provides a collection of pairs ${(a^{(i)}, u^{(i)})}_i$ where $u^{(i)}$
       are the evaluation of the solution operator $\mathcal{G}: a \\to u$
       that maps the diffusion coefficient $a$ into the solution $u$.
        """
    )

    Dict_model = {
        "Convolutional Neural Operator": "CNO",
        "Fourier Neural Operator": "FNO",
    }
    model_to_hyperparams = {
        "CNO": ["default", "best"],
        "FNO": ["default", "best", "best_same_dofs"],
    }

    cols = st.columns(3)
    st.header("Neural operator approximation results on test set")
    with cols[0]:
        ## Selection of the model
        model = Dict_model[
            st.selectbox(
                "Select the model",
                ["Fourier Neural Operator", "Convolutional Neural Operator"],
            )
        ]

    with cols[1]:
        ## Selection of the model
        hyperparams_mode = st.selectbox(
            "Select the hyperparameters mode",
            model_to_hyperparams[model],
        )

    with cols[2]:
        ## Selection of the indexes
        sample_idx = st.number_input(
            f"Index of the sample to plot",
            min_value=0,
            max_value=127,
            value=42,
            step=1,
            key=f"sample_idx",
        )

    ## Read the data
    data = read_data("darcy", model, hyperparams_mode.replace("_", ""))

    ## Plots the selected data
    cols = st.columns([2, 5, 1, 5, 2])
    # Plot of the input
    with cols[1]:
        st.plotly_chart(
            plot_data(
                data["input"],
                sample_idx,
                ylabel="y",
                title="Input function",
            ),
            key="input",
        )

    # Plot of the true output
    with cols[3]:
        st.plotly_chart(
            plot_data(
                data["output"],
                sample_idx,
                ylabel=None,
                title="High-fidelity solution",
            ),
            key="output",
        )

    # Plot of the approximate output
    cols = st.columns([2, 5, 1, 5, 2])
    with cols[3]:
        st.plotly_chart(
            plot_data(
                data["prediction"],
                sample_idx,
                ylabel=None,
                title=f"{model} prediction",
            ),
            key="prediction",
        )

    # Plot of the absolute error
    with cols[1]:
        st.plotly_chart(
            plot_error(
                data,
                sample_idx,
                ylabel=None,
                title=f"Absolute error",
            ),
            key="error",
        )

        ## Error distribution section
    st.header("Error distribution", anchor=False)
    rel_err = compute_relative_error(data)

    cols = st.columns([3, 2])
    with cols[0]:
        st.plotly_chart(
            plot_histogram(
                rel_err, title="Histogram of the relative error distribution", bins=15
            ),
            key="histogram",
        )

    with cols[1]:
        st.plotly_chart(
            plot_swarmplot(
                rel_err, title="Swarmplot of the relative error distribution"
            ),
            key="swarmplot",
        )


if __name__ == "__main__":
    st.set_page_config(
        page_title="HyperNOs",
        layout="wide",
        page_icon=":moyai:",
        menu_items={
            "Report a bug": "https://github.com/MaxGhi8/HyperNOs_website/issues",
            # "About": "sium",
        },
    )
    darcy_page()
