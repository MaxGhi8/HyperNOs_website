import sys

import streamlit as st

sys.path.append("..")
from utilities import (
    read_data,
    plot_data,
    plot_error,
    compute_relative_error,
    plot_histogram,
    plot_swarmplot,
)


def airfoil_page():
    st.title("Flow past airfoil benchmark", anchor=False)

    st.markdown("This benchmark represents the following non-linear PDE")
    st.latex(
        r"""\begin{cases}
                  \frac{\partial u}{\partial t} + \nabla \cdot F(u) = 0 \\
                  u = [\rho,\, \rho v,\, E]^T                           \\
                  F = [\rho v,\, \rho v \otimes v + p\mathbf{I},\, (E + p)v]^T
              \end{cases}"""
    )
    st.markdown(
        """
    with freestream boundary conditions.
    The dataset provides a collection of pairs from the shape of the airfoil to the solution.
    """
    )

    Dict_model = {
        "Convolutional Neural Operator": "CNO",
        "Fourier Neural Operator": "FNO",
    }
    model_to_hyperparams = {
        "CNO": ["default", "best"],
        "FNO": ["default", "best"],
    }

    st.header("Neural operator approximation results on test set")
    cols = st.columns(3)
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
    data = read_data("airfoil", model, hyperparams_mode.replace("_", ""))

    ## Plots the selected data
    cols = st.columns(4)
    # Plot of the input
    with cols[0]:
        st.plotly_chart(
            plot_data(
                data["input"],
                sample_idx,
                ylabel="y",
                title="Input function",
                big_axis=True,
            ),
            key="input",
        )

    # Plot of the true output
    with cols[1]:
        st.plotly_chart(
            plot_data(
                data["output"],
                sample_idx,
                ylabel=None,
                title="High-fidelity solution",
                big_axis=True,
            ),
            key="output",
        )

    # Plot of the approximate output
    with cols[2]:
        st.plotly_chart(
            plot_data(
                data["prediction"],
                sample_idx,
                ylabel=None,
                title=f"{model} prediction",
                big_axis=True,
            ),
            key="prediction",
        )

    # Plot of the absolute error
    with cols[3]:
        st.plotly_chart(
            plot_error(
                data,
                sample_idx,
                ylabel=None,
                title=f"Absolute error",
                big_axis=True,
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
                rel_err, title="Histogram of the relative error distribution"
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
    airfoil_page()
