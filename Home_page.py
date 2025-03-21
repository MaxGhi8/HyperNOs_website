import streamlit as st


# Main page content
def main():

    # Title
    st.title(
        "HyperNOs: Automated and Parallel Library for Neural Operators Research",
        anchor=False,
    )
    # Create a container with custom CSS
    st.markdown(
        """
        <style>
        .profile-name {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 15px;
            font-size: 1.2em;
            font-weight: bold;
            color: var(--text-color);
        }
        .profile-title {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 5px;
            font-size: 0.9em;
            color: var(--text-color);
        }
        .profile-title-secondary {
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            top: -15px;
            margin-bottom: -10px;
            font-size: 0.9em;
            color: var(--text-color);
        }
        .email-button {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 10px;
        }
        .email-button a {
            background-color: #0066cc;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 0.9em;
            transition: background-color 0.3s;
        }
        .email-button a:hover {
            background-color: #0052a3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create columns for the cards
    col1, col2, col3 = st.columns([1, 1, 1], border=False)

    with col2:
        with st.container():

            cols = st.columns([1, 5, 1])
            with cols[1]:
                st.image(
                    "massimiliano_ghiotto.jpg", width=120, use_container_width=True
                )

            st.markdown(
                "<p class='profile-name'>Massimiliano Ghiotto</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p class='profile-title'>University of Pavia</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="email-button">
                    <a href="mailto:massimiliano.ghiotto01@universitadipavia.it">
                        <span class="email-emoji">🖂</span> Contact Me
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Description
    st.header("Project Description", anchor=False)
    st.markdown(
        """
        This page is associated to HyperNOs library, a PyTorch library designed to streamline and automate the process of exploring neural operators,
        with a special focus on hyperparameter optimization for comprehensive and exhaustive exploration. 
        Indeed HyperNOs takes advantage of state-of-the-art optimization algorithms and parallel computing implemented in the Ray-tune
        library to efficiently explore the hyperparameter space of neural operators.
        I also implement many useful functionalities for studying neural operators with a user-friendly interface,
        such as the possibility to train the model with with a fixed number of parameters or to train the model with multiple datasets and different resolutions.
        I integrate Fourier neural operators and convolutional neural operators in our library achieving state of the art results on a lot of representative benchmarks,
        to demonstrate the capabilities of HyperNOs to handle real datasets and modern architectures.
        The library is designed to be easy to use with the provided model and datasets,
        but also to be easily extended to use new datasets and custom neural operator architectures.
        
        In this page I collect the numerical results of the experiments conducted along my article,
        in particular for every trained model here I collect here the approximated solutions across the test set. 
        """
    )

    # FNO video
    st.header("Fourier Neural Operator animation", anchor=False)
    st.markdown(
        """
    In this video, I show the architecture of the Fourier Neural Operator for the two-dimensional case, that is the case of interest for our work.
    The inputs and outputs are taken from the Darcy flow dataset, mapping from the diffusion coefficient to the approximated PDE solution.
    """
    )
    st.video("FNO_architecture_2d.mp4", loop=True, autoplay=True)

    # FNO image
    st.header("Fourier Neural Operator architecture", anchor=False)
    st.image("FNO_arc.png")

    # CNO image
    st.header("Convolutional Neural Operator architecture", anchor=False)
    st.image("CNO_arc.png")


if __name__ == "__main__":
    st.set_page_config(
        page_title="HyperNOs",
        page_icon=":moyai:",
        menu_items={
            "Report a bug": "https://github.com/MaxGhi8/HyperNOs_website/issues",
            # "About": "sium",
        },
    )
    main()
