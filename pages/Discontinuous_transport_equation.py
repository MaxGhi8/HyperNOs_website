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


def disc_tran_page():
    st.title("Discontinuous transport equation benchmark", anchor=False)

    st.markdown("This benchmark represents a non-linear PDE, given by")
    st.latex(
        r"""\begin{cases}
                  \frac{\partial u}{\partial t}  + v\cdot \nabla u= 0 \quad & in \ D\times (0, T) \\
                  u(x, 0) = f(x)                                            & in\ D               \\
                  v = [0.2, \, 0.2]^T
              \end{cases}"""
    )
    st.markdown(
        """
    The dataset provides a collection of pairs ${(f^{(i)}, u^{(i)})}_i$ where $u^{(i)}$
    are the evaluation of the solution operator $\mathcal{G}: f \\to u(\cdot, T)$
    that maps the initial condition $f$ into the solution at the final time $T=1$.
    This is with discontinuous initial data.
    """
    )

    Dict_model = {
        "Convolutional Neural Operator": "CNO",
        "Fourier Neural Operator": "FNO",
    }
    model_to_hyperparams = {
        "CNO": ["default", "best", "best_same_dofs"],
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
            max_value=255,
            value=42,
            step=1,
            key=f"sample_idx",
        )

    ## Read the data
    data = read_data("disc_tran", model, hyperparams_mode.replace("_", ""))

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
    disc_tran_page()
