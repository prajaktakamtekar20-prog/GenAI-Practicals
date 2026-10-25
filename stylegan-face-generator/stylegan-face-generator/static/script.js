const generationMode =
    document.getElementById("generationMode");

const skinTone =
    document.getElementById("skinTone");

const hairColor =
    document.getElementById("hairColor");

const generateBtn =
    document.getElementById("generateBtn");

const generatedImage =
    document.getElementById("generatedImage");

const placeholder =
    document.getElementById("placeholder");

const statusText =
    document.getElementById("status");


// ============================================================
// CHANGE MODE
// ============================================================

generationMode.addEventListener(
    "change",
    function () {

        if (this.value === "normal") {

            generateBtn.innerText =
                "Generate Face";

        }

        else if (this.value === "mix") {

            generateBtn.innerText =
                "Generate Style Mixed Face";

        }

        else if (this.value === "noise") {

            generateBtn.innerText =
                "Generate Noise Variation";

        }

    }
);


// ============================================================
// GENERATE
// ============================================================

generateBtn.addEventListener(
    "click",
    async function () {

        generateBtn.disabled = true;

        statusText.innerText =
            "Generating face...";


        const mode =
            generationMode.value;

        const skin =
            skinTone.value;

        const hair =
            hairColor.value;


        let endpoint =
            "/generate";


        // Select backend feature

        if (mode === "mix") {

            endpoint =
                "/style-mix";

        }

        else if (mode === "noise") {

            endpoint =
                "/noise-variation";

        }


        // Add feature parameters

        endpoint +=
            "?skin=" +
            encodeURIComponent(skin) +
            "&hair=" +
            encodeURIComponent(hair);


        try {

            const response =
                await fetch(endpoint);


            if (!response.ok) {

                throw new Error(
                    "Server error"
                );

            }


            const data =
                await response.json();


            if (data.success) {

                generatedImage.src =
                    data.image +
                    "?t=" +
                    new Date().getTime();


                generatedImage.style.display =
                    "block";


                placeholder.style.display =
                    "none";


                statusText.innerText =
                    "Face generated successfully!";

            }

            else {

                statusText.innerText =
                    "Generation failed.";

            }

        }

        catch (error) {

            console.error(error);

            statusText.innerText =
                "Error connecting to Flask server.";

        }


        generateBtn.disabled = false;

    }
);