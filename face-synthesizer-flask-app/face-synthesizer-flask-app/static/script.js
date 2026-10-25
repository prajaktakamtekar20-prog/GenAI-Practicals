const generateButton =
    document.getElementById("generateBtn");

const loading =
    document.getElementById("loading");

const generatedImage =
    document.getElementById("generatedImage");


generateButton.addEventListener("click", async function () {

    generateButton.disabled = true;

    loading.style.display = "block";

    generatedImage.style.display = "none";


    try {

        const response =
            await fetch("/generate");

        const data =
            await response.json();


        if (data.success) {

            generatedImage.src =
                data.image + "?t=" + new Date().getTime();

            generatedImage.style.display = "inline-block";

        }

    }

    catch (error) {

        console.error(error);

        alert("Error generating face.");

    }


    loading.style.display = "none";

    generateButton.disabled = false;

});