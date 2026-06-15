/**
 * DNA Data Encoding — Interactive Demo Engine
 *
 * Runs the full encoding/decoding pipeline entirely in the browser.
 * No server required. Loads pre-computed training results from demo_data.json.
 *
 * All variable and function names use full words — no abbreviations.
 */

// ============================================================
// ENCODING CONSTANTS
// ============================================================
const NUCLEOTIDES = ["A", "C", "G", "T"];
const DIBITS = [[0,0], [0,1], [1,0], [1,1]];
const DIBIT_LABELS = ["00", "01", "10", "11"];

// Generate all 24 permutation mappings
function generateAllMappings() {
    const permutations = [];
    const elements = [0, 1, 2, 3];

    function permute(array, current) {
        if (current.length === 4) {
            permutations.push([...current]);
            return;
        }
        for (let index = 0; index < array.length; index++) {
            const remaining = [...array];
            remaining.splice(index, 1);
            permute(remaining, [...current, array[index]]);
        }
    }
    permute(elements, []);

    return permutations.map(permutation => {
        const mapping = {};
        for (let index = 0; index < 4; index++) {
            mapping[DIBIT_LABELS[index]] = NUCLEOTIDES[permutation[index]];
        }
        return mapping;
    });
}

const ALL_MAPPINGS = generateAllMappings();
const NUCLEOTIDE_COLOURS = { A: "#2ecc71", C: "#3498db", G: "#f1c40f", T: "#e74c3c" };

// ============================================================
// DEMO DATA (loaded from JSON if available)
// ============================================================
let demoData = null;

async function loadDemoData() {
    try {
        const response = await fetch("demo_data.json");
        if (response.ok) {
            demoData = await response.json();
            console.log("Demo data loaded successfully.");
            renderTrainingCharts();
            renderResultsCharts();
            renderResultsTable();
            renderGeneralisationTest();
            renderRedundancyAnalysisChart();
            renderAblationChart();
        }
    } catch (error) {
        console.log("No demo_data.json found. Running in standalone mode.");
    }
}

// ============================================================
// TEXT TO BINARY CONVERSION
// ============================================================

function textToBinary(text) {
    const binaryBits = [];
    for (let characterIndex = 0; characterIndex < text.length; characterIndex++) {
        const characterCode = text.charCodeAt(characterIndex);
        for (let bitPosition = 7; bitPosition >= 0; bitPosition--) {
            binaryBits.push((characterCode >> bitPosition) & 1);
        }
    }
    // pad to even length for dibit encoding
    while (binaryBits.length % 2 !== 0) {
        binaryBits.push(0);
    }
    return binaryBits;
}

function binaryToText(binaryBits) {
    const characters = [];
    for (let index = 0; index < binaryBits.length; index += 8) {
        let characterCode = 0;
        for (let bitPosition = 0; bitPosition < 8 && index + bitPosition < binaryBits.length; bitPosition++) {
            characterCode = (characterCode << 1) | binaryBits[index + bitPosition];
        }
        if (characterCode > 0 && characterCode < 128) {
            characters.push(String.fromCharCode(characterCode));
        }
    }
    return characters.join("");
}

function binaryToDibits(binaryBits) {
    const dibits = [];
    for (let index = 0; index < binaryBits.length; index += 2) {
        dibits.push("" + binaryBits[index] + binaryBits[index + 1]);
    }
    return dibits;
}

function computeDibitDistribution(dibits) {
    const counts = { "00": 0, "01": 0, "10": 0, "11": 0 };
    for (const dibit of dibits) {
        counts[dibit]++;
    }
    const total = dibits.length || 1;
    return {
        "00": counts["00"] / total,
        "01": counts["01"] / total,
        "10": counts["10"] / total,
        "11": counts["11"] / total,
    };
}

// ============================================================
// DNA ENCODING / DECODING
// ============================================================

function encodeDibitsToNucleotides(dibits, mapping) {
    return dibits.map(dibit => mapping[dibit] || "A").join("");
}

function decodeNucleotidesToDibits(dnaSequence, mapping) {
    const reverseMapping = {};
    for (const [dibit, nucleotide] of Object.entries(mapping)) {
        reverseMapping[nucleotide] = dibit;
    }
    const dibits = [];
    for (const nucleotide of dnaSequence) {
        dibits.push(reverseMapping[nucleotide] || "00");
    }
    return dibits;
}

function dibitsToIntegers(dibits) {
    const bits = [];
    for (const dibit of dibits) {
        bits.push(parseInt(dibit[0]), parseInt(dibit[1]));
    }
    return bits;
}

// ============================================================
// MAPPING SELECTION (simple heuristic in absence of trained model)
// ============================================================

function selectBestMapping(dibitDistribution) {
    // If we have a trained policy table, use it
    if (demoData && demoData.policy_lookup_table && demoData.policy_lookup_table.length > 0) {
        // Find closest match in policy table
        let bestMatch = demoData.policy_lookup_table[0];
        let bestDistance = Infinity;
        for (const entry of demoData.policy_lookup_table) {
            const distance = Math.abs(entry.dibit_distribution[0] - dibitDistribution["00"])
                           + Math.abs(entry.dibit_distribution[1] - dibitDistribution["01"])
                           + Math.abs(entry.dibit_distribution[2] - dibitDistribution["10"])
                           + Math.abs(entry.dibit_distribution[3] - dibitDistribution["11"]);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestMatch = entry;
            }
        }
        return bestMatch.primary_mapping_index;
    }

    // Fallback: GC-balancing heuristic
    // For each mapping, compute expected GC content and pick closest to 0.5
    let bestIndex = 0;
    let bestScore = Infinity;
    for (let mappingIndex = 0; mappingIndex < ALL_MAPPINGS.length; mappingIndex++) {
        const mapping = ALL_MAPPINGS[mappingIndex];
        let gcEstimate = 0;
        for (const dibit of DIBIT_LABELS) {
            const nucleotide = mapping[dibit];
            if (nucleotide === "G" || nucleotide === "C") {
                gcEstimate += dibitDistribution[dibit];
            }
        }
        const score = Math.abs(gcEstimate - 0.5);
        if (score < bestScore) {
            bestScore = score;
            bestIndex = mappingIndex;
        }
    }
    return bestIndex;
}

// ============================================================
// ERROR SIMULATION (JavaScript port)
// ============================================================

function simulateErrors(dnaSequence, substitutionRate, insertionRate, deletionRate) {
    const result = [];
    const errorDetails = []; // for visualization
    const statistics = { substitutions: 0, insertions: 0, deletions: 0 };

    for (let position = 0; position < dnaSequence.length; position++) {
        let nucleotide = dnaSequence[position];
        const randomValue = Math.random();

        if (randomValue < deletionRate) {
            statistics.deletions++;
            errorDetails.push({ position, type: "deletion", original: nucleotide });
            continue;
        }

        if (randomValue < deletionRate + substitutionRate) {
            const alternatives = NUCLEOTIDES.filter(n => n !== nucleotide);
            const newNucleotide = alternatives[Math.floor(Math.random() * alternatives.length)];
            errorDetails.push({ position, type: "substitution", original: nucleotide, mutated: newNucleotide });
            nucleotide = newNucleotide;
            statistics.substitutions++;
        }

        result.push(nucleotide);

        if (Math.random() < insertionRate) {
            const insertedNucleotide = NUCLEOTIDES[Math.floor(Math.random() * 4)];
            result.push(insertedNucleotide);
            errorDetails.push({ position, type: "insertion", inserted: insertedNucleotide });
            statistics.insertions++;
        }
    }

    return {
        mutatedSequence: result.join(""),
        statistics,
        errorDetails,
    };
}

// ============================================================
// MAJORITY VOTING
// ============================================================

function majorityVote(copies) {
    if (copies.length === 0) return "";
    if (copies.length === 1) return copies[0];

    const lengths = copies.map(copy => copy.length);
    const medianLength = lengths.sort((a, b) => a - b)[Math.floor(lengths.length / 2)];

    const consensus = [];
    for (let position = 0; position < medianLength; position++) {
        const votes = {};
        for (const copy of copies) {
            if (position < copy.length) {
                const nucleotide = copy[position];
                votes[nucleotide] = (votes[nucleotide] || 0) + 1;
            }
        }
        let bestNucleotide = "A";
        let bestCount = 0;
        for (const [nucleotide, count] of Object.entries(votes)) {
            if (count > bestCount) {
                bestCount = count;
                bestNucleotide = nucleotide;
            }
        }
        consensus.push(bestNucleotide);
    }
    return consensus.join("");
}

// ============================================================
// ACCURACY CALCULATION
// ============================================================

function computeBitAccuracy(original, decoded) {
    const minimumLength = Math.min(original.length, decoded.length);
    const maximumLength = Math.max(original.length, decoded.length);
    if (maximumLength === 0) return 1.0;
    let matches = 0;
    for (let index = 0; index < minimumLength; index++) {
        if (original[index] === decoded[index]) matches++;
    }
    return matches / maximumLength;
}

function computeGCContent(dnaSequence) {
    let gcCount = 0;
    for (const nucleotide of dnaSequence) {
        if (nucleotide === "G" || nucleotide === "C") gcCount++;
    }
    return gcCount / (dnaSequence.length || 1);
}

function computeMaxHomopolymer(dnaSequence) {
    if (dnaSequence.length === 0) return 0;
    let maxRun = 1, currentRun = 1;
    for (let index = 1; index < dnaSequence.length; index++) {
        if (dnaSequence[index] === dnaSequence[index - 1]) {
            currentRun++;
            maxRun = Math.max(maxRun, currentRun);
        } else {
            currentRun = 1;
        }
    }
    return maxRun;
}

// ============================================================
// DNA STRAND RENDERER
// ============================================================

function renderDNAStrand(container, dnaSequence, errorDetails = null) {
    container.innerHTML = "";
    const maxDisplay = 200;
    const displaySequence = dnaSequence.substring(0, maxDisplay);

    // Build error map by output position
    const errorMap = {};
    if (errorDetails) {
        for (const error of errorDetails) {
            errorMap[error.position] = error;
        }
    }

    for (let index = 0; index < displaySequence.length; index++) {
        const nucleotide = displaySequence[index];
        const span = document.createElement("span");
        span.className = `nt nt-${nucleotide}`;
        span.textContent = nucleotide;

        if (errorMap[index]) {
            if (errorMap[index].type === "substitution") {
                span.classList.add("nt-error-substitution");
                span.title = `Substitution: ${errorMap[index].original} → ${nucleotide}`;
            } else if (errorMap[index].type === "insertion") {
                span.classList.add("nt-error-insertion");
                span.title = `Insertion after position ${index}`;
            }
        }

        container.appendChild(span);
    }

    if (dnaSequence.length > maxDisplay) {
        const ellipsis = document.createElement("span");
        ellipsis.textContent = ` ... (${dnaSequence.length} total nucleotides)`;
        ellipsis.style.color = "var(--text-muted)";
        ellipsis.style.fontSize = "0.75rem";
        container.appendChild(ellipsis);
    }
}

// ============================================================
// PIPELINE EXECUTION
// ============================================================

let pipelineRunning = false;

async function runPipeline() {
    if (pipelineRunning) return;
    pipelineRunning = true;

    const inputText = document.getElementById("demo-text-input").value.trim();
    if (!inputText) {
        alert("Please enter some text to encode.");
        pipelineRunning = false;
        return;
    }

    const numberOfCopies = parseInt(document.getElementById("copy-count-select").value);

    // Reset all steps
    document.querySelectorAll(".pipeline-step").forEach(step => {
        step.classList.remove("active", "completed");
        step.querySelector(".step-status").textContent = "Waiting";
    });

    // === Step 1: Binary Conversion ===
    await activateStep("step-binary", "Processing");
    const binaryBits = textToBinary(inputText);
    const dibits = binaryToDibits(binaryBits);

    // Render binary
    const binaryDisplay = document.getElementById("binary-display");
    binaryDisplay.innerHTML = "<strong>Binary:</strong> " +
        binaryBits.map(bit => `<span class="bit-${bit}">${bit}</span>`).join("");

    // Render dibits
    const dibitDisplay = document.getElementById("dibit-display");
    dibitDisplay.innerHTML = "<strong>Dibits:</strong> " +
        dibits.map(dibit => `<span class="dibit dibit-${dibit}">${dibit}</span>`).join(" ");

    await delay(600);
    completeStep("step-binary");

    // === Step 2: Data Analysis ===
    await activateStep("step-analysis", "Analysing");
    const distribution = computeDibitDistribution(dibits);
    renderDibitChart(distribution);
    await delay(600);
    completeStep("step-analysis");

    // === Step 3: Encoding Selection ===
    await activateStep("step-encoding", "Selecting");
    const mappingIndex = selectBestMapping(distribution);
    const selectedMapping = ALL_MAPPINGS[mappingIndex];
    renderSelectedMapping(mappingIndex, selectedMapping);
    await delay(600);
    completeStep("step-encoding");

    // === Step 4: DNA Encoding ===
    await activateStep("step-dna", "Encoding");
    const dnaSequence = encodeDibitsToNucleotides(dibits, selectedMapping);
    renderDNAStrand(document.getElementById("dna-strand-original"), dnaSequence);

    const gcValue = computeGCContent(dnaSequence);
    const maxHomopolymerValue = computeMaxHomopolymer(dnaSequence);
    document.getElementById("dna-metrics").innerHTML = `
        <span class="dna-metric"><strong>Length:</strong> ${dnaSequence.length} nucleotides</span>
        <span class="dna-metric"><strong>GC Content:</strong> ${(gcValue * 100).toFixed(1)}%</span>
        <span class="dna-metric"><strong>Max Homopolymer:</strong> ${maxHomopolymerValue}</span>
    `;
    await delay(600);
    completeStep("step-dna");

    // === Step 5: Multi-Copy Storage ===
    await activateStep("step-copies", "Storing");
    const copiesContainer = document.getElementById("copies-container");
    copiesContainer.innerHTML = "";
    for (let copyIndex = 0; copyIndex < numberOfCopies; copyIndex++) {
        const copyDiv = document.createElement("div");
        copyDiv.innerHTML = `<span class="copy-label">Copy ${copyIndex + 1}</span>`;
        const strandDiv = document.createElement("div");
        strandDiv.className = "dna-strand";
        renderDNAStrand(strandDiv, dnaSequence);
        copyDiv.appendChild(strandDiv);
        copiesContainer.appendChild(copyDiv);
    }
    await delay(600);
    completeStep("step-copies");

    // === Step 6: Error Simulation ===
    // NOTE: The pipeline demo uses substitution-only errors because
    // insertions and deletions shift nucleotide positions, which breaks
    // position-wise majority voting. The Error Laboratory section below
    // lets you experiment with all three error types independently.
    await activateStep("step-errors", "Simulating Errors");
    const mutatedCopies = [];
    const errorCopiesContainer = document.getElementById("error-copies");
    errorCopiesContainer.innerHTML = "";

    const DEMO_SUBSTITUTION_RATE = 0.04;  // 4% substitution rate (no indels)

    for (let copyIndex = 0; copyIndex < numberOfCopies; copyIndex++) {
        const { mutatedSequence, statistics, errorDetails } = simulateErrors(
            dnaSequence, DEMO_SUBSTITUTION_RATE, 0.0, 0.0
        );
        mutatedCopies.push(mutatedSequence);

        const copyDiv = document.createElement("div");
        copyDiv.innerHTML = `<span class="copy-label">Copy ${copyIndex + 1} — ` +
            `<span style="color:var(--error-substitution)">${statistics.substitutions} substitutions</span> ` +
            `<span style="color:var(--text-muted)">(${dnaSequence.length} nucleotides, ` +
            `${(statistics.substitutions / dnaSequence.length * 100).toFixed(1)}% error rate)</span></span>`;
        const strandDiv = document.createElement("div");
        strandDiv.className = "dna-strand";
        renderDNAStrand(strandDiv, mutatedSequence, errorDetails);
        copyDiv.appendChild(strandDiv);
        errorCopiesContainer.appendChild(copyDiv);
    }
    await delay(800);
    completeStep("step-errors");

    // === Step 7: Majority Voting ===
    await activateStep("step-voting", "Voting");
    const consensusSequence = majorityVote(mutatedCopies);
    const votingDisplay = document.getElementById("voting-display");
    votingDisplay.innerHTML = "";

    // Show each copy + consensus (limited to first 60 positions)
    const displayLength = Math.min(60, consensusSequence.length);
    for (let copyIndex = 0; copyIndex < numberOfCopies; copyIndex++) {
        const row = document.createElement("div");
        row.className = "vote-row";
        row.innerHTML = `<span class="vote-label">Copy ${copyIndex + 1}</span>`;
        for (let position = 0; position < displayLength; position++) {
            const nucleotide = mutatedCopies[copyIndex][position] || "-";
            const span = document.createElement("span");
            span.className = `vote-nt nt-${nucleotide}`;
            span.textContent = nucleotide;
            row.appendChild(span);
        }
        votingDisplay.appendChild(row);
    }

    // Consensus row
    const consensusRow = document.createElement("div");
    consensusRow.className = "vote-row";
    consensusRow.innerHTML = `<span class="vote-label consensus-label">Consensus</span>`;
    for (let position = 0; position < displayLength; position++) {
        const nucleotide = consensusSequence[position] || "-";
        const span = document.createElement("span");
        span.className = `vote-nt nt-${nucleotide} vote-winner`;
        span.textContent = nucleotide;
        consensusRow.appendChild(span);
    }
    votingDisplay.appendChild(consensusRow);

    if (consensusSequence.length > displayLength) {
        const moreText = document.createElement("div");
        moreText.style.cssText = "font-size:0.75rem;color:var(--text-muted);margin-top:0.5rem;";
        moreText.textContent = `Showing ${displayLength} of ${consensusSequence.length} positions`;
        votingDisplay.appendChild(moreText);
    }

    await delay(600);
    completeStep("step-voting");

    // === Step 8: Decoding ===
    await activateStep("step-decode", "Decoding");
    const decodedDibits = decodeNucleotidesToDibits(consensusSequence, selectedMapping);
    const decodedBits = dibitsToIntegers(decodedDibits);
    const decodedText = binaryToText(decodedBits);

    // Single copy accuracy: average across ALL copies (fairer comparison)
    let totalSingleAccuracy = 0;
    let worstSingleAccuracy = 100;
    let bestSingleAccuracy = 0;
    let singleCopyDecodedText = "";
    for (let copyIndex = 0; copyIndex < numberOfCopies; copyIndex++) {
        const copyDibits = decodeNucleotidesToDibits(mutatedCopies[copyIndex], selectedMapping);
        const copyBits = dibitsToIntegers(copyDibits);
        const copyAccuracy = computeBitAccuracy(binaryBits, copyBits) * 100;
        totalSingleAccuracy += copyAccuracy;
        if (copyAccuracy < worstSingleAccuracy) worstSingleAccuracy = copyAccuracy;
        if (copyAccuracy > bestSingleAccuracy) bestSingleAccuracy = copyAccuracy;
        if (copyIndex === 0) singleCopyDecodedText = binaryToText(copyBits);
    }
    const averageSingleAccuracy = totalSingleAccuracy / numberOfCopies;

    // Multi-copy accuracy (consensus)
    const multiCopyAccuracy = computeBitAccuracy(binaryBits, decodedBits) * 100;
    const improvement = multiCopyAccuracy - averageSingleAccuracy;

    const decodeResults = document.getElementById("decode-results");
    // Highlight mismatched characters
    function highlightMismatches(original, decoded) {
        let result = "";
        const maxLen = Math.max(original.length, decoded.length);
        for (let i = 0; i < maxLen; i++) {
            const origChar = original[i] || "";
            const decChar = decoded[i] || "";
            if (origChar !== decChar) {
                // If it's a non-matching character, highlight it
                // Handle cases where the decoded string is longer or shorter (though with fixed length this shouldn't happen much)
                const displayChar = decChar === "" ? "&empty;" : escapeHTML(decChar);
                result += `<span class="mismatch-highlight">${displayChar}</span>`;
            } else {
                result += escapeHTML(decChar);
            }
        }
        return result;
    }

    const singleCopyHighlighted = highlightMismatches(inputText, singleCopyDecodedText);
    const multiCopyHighlighted = highlightMismatches(inputText, decodedText);

    decodeResults.innerHTML = `
        <div style="margin-bottom:1rem;">
            <strong>Original Text:</strong>
            <div class="decode-text" style="opacity:0.6;">${escapeHTML(inputText)}</div>
        </div>
        <div style="margin-bottom:1rem;">
            <strong>Single Copy Decoded (copy 1):</strong>
            <div class="decode-text" style="border-left:3px solid var(--error-substitution);">${singleCopyHighlighted}</div>
        </div>
        <div style="margin-bottom:1rem;">
            <strong>Multi-Copy Consensus Decoded (${numberOfCopies} copies):</strong>
            <div class="decode-text" style="border-left:3px solid #2ecc71;">${multiCopyHighlighted}</div>
        </div>
        <div class="accuracy-comparison">
            <div class="accuracy-card">
                <div class="accuracy-label">Average Single Copy</div>
                <div class="accuracy-value ${averageSingleAccuracy > 90 ? 'good' : averageSingleAccuracy > 70 ? 'medium' : 'poor'}">${averageSingleAccuracy.toFixed(1)}%</div>
                <div class="accuracy-label">Without redundancy (best: ${bestSingleAccuracy.toFixed(1)}%, worst: ${worstSingleAccuracy.toFixed(1)}%)</div>
            </div>
            <div class="accuracy-card">
                <div class="accuracy-label">Multi-Copy Consensus (${numberOfCopies} copies)</div>
                <div class="accuracy-value ${multiCopyAccuracy > 90 ? 'good' : multiCopyAccuracy > 70 ? 'medium' : 'poor'}">${multiCopyAccuracy.toFixed(1)}%</div>
                <div class="accuracy-label">With majority voting</div>
            </div>
            <div class="accuracy-card">
                <div class="accuracy-label">Improvement</div>
                <div class="accuracy-value ${improvement >= 0 ? 'good' : 'poor'}">${improvement >= 0 ? '+' : ''}${improvement.toFixed(1)}%</div>
                <div class="accuracy-label">From redundancy</div>
            </div>
        </div>
    `;

    completeStep("step-decode");
    pipelineRunning = false;
}

// ============================================================
// PIPELINE STEP HELPERS
// ============================================================

function activateStep(stepId, statusText) {
    return new Promise(resolve => {
        const step = document.getElementById(stepId);
        step.classList.add("active");
        step.querySelector(".step-status").textContent = statusText;
        step.scrollIntoView({ behavior: "smooth", block: "nearest" });
        setTimeout(resolve, 200);
    });
}

function completeStep(stepId) {
    const step = document.getElementById(stepId);
    step.classList.remove("active");
    step.classList.add("completed");
    step.querySelector(".step-status").textContent = "Complete";
}

function delay(milliseconds) {
    return new Promise(resolve => setTimeout(resolve, milliseconds));
}

function escapeHTML(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================
// DIBIT CHART
// ============================================================

function renderDibitChart(distribution) {
    const container = document.getElementById("dibit-chart");
    container.innerHTML = "";

    const colours = { "00": "#2ecc71", "01": "#3498db", "10": "#f1c40f", "11": "#e74c3c" };
    const labels = { "00": "00→A", "01": "01→C", "10": "10→G", "11": "11→T" };
    const maxValue = Math.max(...Object.values(distribution), 0.01);

    for (const dibit of DIBIT_LABELS) {
        const wrapper = document.createElement("div");
        wrapper.className = "dibit-bar-wrapper";

        const value = document.createElement("div");
        value.className = "dibit-bar-value";
        value.textContent = (distribution[dibit] * 100).toFixed(1) + "%";
        value.style.color = colours[dibit];

        const bar = document.createElement("div");
        bar.className = "dibit-bar";
        bar.style.backgroundColor = colours[dibit];
        bar.style.height = (distribution[dibit] / maxValue * 70) + "%";
        bar.style.marginTop = "auto";

        const label = document.createElement("div");
        label.className = "dibit-bar-label";
        label.textContent = labels[dibit];

        wrapper.appendChild(value);
        wrapper.appendChild(bar);
        wrapper.appendChild(label);
        container.appendChild(wrapper);
    }
}

// ============================================================
// SELECTED MAPPING RENDERER
// ============================================================

function renderSelectedMapping(mappingIndex, mapping) {
    const container = document.getElementById("selected-mapping");
    container.innerHTML = `
        <div class="mapping-title">Reinforcement Learning Agent Selected: Mapping #${mappingIndex}</div>
        ${DIBIT_LABELS.map(dibit => {
            const nucleotide = mapping[dibit];
            return `<div class="mapping-row">
                <span class="dibit dibit-${dibit}">${dibit}</span>
                <span style="color:var(--text-muted)">→</span>
                <span class="nt nt-${nucleotide}" style="display:inline-block;width:24px;height:24px;text-align:center;line-height:24px;">${nucleotide}</span>
                <span style="color:var(--text-muted);font-size:0.8rem;">${{A:"Adenine",C:"Cytosine",G:"Guanine",T:"Thymine"}[nucleotide]}</span>
            </div>`;
        }).join("")}
    `;
}

// ============================================================
// ALL 24 MAPPINGS GRID
// ============================================================

function renderMappingsGrid() {
    const grid = document.getElementById("mappings-grid");
    if (!grid) return;

    for (let index = 0; index < ALL_MAPPINGS.length; index++) {
        const mapping = ALL_MAPPINGS[index];
        const card = document.createElement("div");
        card.className = "mapping-card";

        let nucleotideSpans = "";
        for (const dibit of DIBIT_LABELS) {
            const nucleotide = mapping[dibit];
            nucleotideSpans += `<span class="mini-nt" style="background:${NUCLEOTIDE_COLOURS[nucleotide]}">${nucleotide}</span>`;
        }

        card.innerHTML = `
            <div class="mapping-index">#${index}</div>
            <div class="mapping-nucleotides">${nucleotideSpans}</div>
        `;
        grid.appendChild(card);
    }
}

// ============================================================
// CANVAS CHART ENGINE
// ============================================================

function drawLineChart(canvasId, datasets, options = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const context = canvas.getContext("2d");

    // Set actual pixel dimensions
    const displayWidth = canvas.parentElement.clientWidth - 40;
    const displayHeight = options.height || 300;
    canvas.width = displayWidth * 2;
    canvas.height = displayHeight * 2;
    canvas.style.width = displayWidth + "px";
    canvas.style.height = displayHeight + "px";
    context.scale(2, 2);

    const padding = { top: 30, right: 30, bottom: 40, left: 60 };
    const chartWidth = displayWidth - padding.left - padding.right;
    const chartHeight = displayHeight - padding.top - padding.bottom;

    // Clear
    context.fillStyle = "rgba(17,24,39,0.5)";
    context.fillRect(0, 0, displayWidth, displayHeight);

    if (!datasets || datasets.length === 0 || datasets.every(dataset => dataset.data.length === 0)) {
        context.fillStyle = "#64748b";
        context.font = "14px Inter";
        context.textAlign = "center";
        context.fillText("No training data available yet. Run the Colab notebook first.",
                        displayWidth / 2, displayHeight / 2);
        return;
    }

    // Compute bounds
    let minValue = Infinity, maxValue = -Infinity, maxLength = 0;
    for (const dataset of datasets) {
        for (const value of dataset.data) {
            if (value < minValue) minValue = value;
            if (value > maxValue) maxValue = value;
        }
        if (dataset.data.length > maxLength) maxLength = dataset.data.length;
    }
    const valueRange = maxValue - minValue || 1;
    minValue -= valueRange * 0.05;
    maxValue += valueRange * 0.05;

    // Grid
    context.strokeStyle = "rgba(148,163,184,0.1)";
    context.lineWidth = 0.5;
    for (let gridLine = 0; gridLine <= 5; gridLine++) {
        const gridY = padding.top + (gridLine / 5) * chartHeight;
        context.beginPath();
        context.moveTo(padding.left, gridY);
        context.lineTo(padding.left + chartWidth, gridY);
        context.stroke();

        const gridValue = maxValue - (gridLine / 5) * (maxValue - minValue);
        context.fillStyle = "#64748b";
        context.font = "10px JetBrains Mono";
        context.textAlign = "right";
        context.fillText(gridValue.toFixed(3), padding.left - 8, gridY + 3);
    }

    // X-axis labels
    context.fillStyle = "#64748b";
    context.font = "10px JetBrains Mono";
    context.textAlign = "center";
    for (let tick = 0; tick <= 5; tick++) {
        const tickIndex = Math.floor((tick / 5) * maxLength);
        const tickX = padding.left + (tick / 5) * chartWidth;
        context.fillText(tickIndex.toString(), tickX, displayHeight - 10);
    }

    // Draw lines
    for (const dataset of datasets) {
        if (dataset.data.length === 0) continue;
        context.strokeStyle = dataset.colour || "#6366f1";
        context.lineWidth = dataset.lineWidth || 1.5;
        context.globalAlpha = dataset.alpha || 1;
        context.beginPath();

        const step = Math.max(1, Math.floor(dataset.data.length / chartWidth));
        for (let index = 0; index < dataset.data.length; index += step) {
            const x = padding.left + (index / maxLength) * chartWidth;
            const y = padding.top + ((maxValue - dataset.data[index]) / (maxValue - minValue)) * chartHeight;
            if (index === 0) context.moveTo(x, y);
            else context.lineTo(x, y);
        }
        context.stroke();
        context.globalAlpha = 1;

        // Label
        if (dataset.label) {
            const lastValue = dataset.data[dataset.data.length - 1];
            const lastX = padding.left + chartWidth;
            const lastY = padding.top + ((maxValue - lastValue) / (maxValue - minValue)) * chartHeight;
            context.fillStyle = dataset.colour || "#6366f1";
            context.font = "10px Inter";
            context.textAlign = "left";
        }
    }

    // Axis labels
    if (options.xLabel) {
        context.fillStyle = "#94a3b8";
        context.font = "12px Inter";
        context.textAlign = "center";
        context.fillText(options.xLabel, padding.left + chartWidth / 2, displayHeight - 2);
    }
}

function drawBarChart(canvasId, groups, options = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const context = canvas.getContext("2d");

    const displayWidth = canvas.parentElement.clientWidth - 40;
    const displayHeight = options.height || 300;
    canvas.width = displayWidth * 2;
    canvas.height = displayHeight * 2;
    canvas.style.width = displayWidth + "px";
    canvas.style.height = displayHeight + "px";
    context.scale(2, 2);

    const padding = { top: 30, right: 30, bottom: 60, left: 60 };
    const chartWidth = displayWidth - padding.left - padding.right;
    const chartHeight = displayHeight - padding.top - padding.bottom;

    context.fillStyle = "rgba(17,24,39,0.5)";
    context.fillRect(0, 0, displayWidth, displayHeight);

    if (!groups || groups.length === 0) {
        context.fillStyle = "#64748b";
        context.font = "14px Inter";
        context.textAlign = "center";
        context.fillText("No results data available yet.", displayWidth / 2, displayHeight / 2);
        return;
    }

    // Compute max value
    let maxValue = 0;
    for (const group of groups) {
        for (const bar of group.bars) {
            if (bar.value > maxValue) maxValue = bar.value;
        }
    }
    maxValue *= 1.15;

    const groupWidth = chartWidth / groups.length;
    const barsPerGroup = groups[0].bars.length;
    const barWidth = (groupWidth * 0.7) / barsPerGroup;
    const groupPadding = groupWidth * 0.15;

    // Grid
    context.strokeStyle = "rgba(148,163,184,0.1)";
    context.lineWidth = 0.5;
    for (let gridLine = 0; gridLine <= 5; gridLine++) {
        const gridY = padding.top + (gridLine / 5) * chartHeight;
        context.beginPath();
        context.moveTo(padding.left, gridY);
        context.lineTo(padding.left + chartWidth, gridY);
        context.stroke();

        const gridValue = maxValue - (gridLine / 5) * maxValue;
        context.fillStyle = "#64748b";
        context.font = "10px JetBrains Mono";
        context.textAlign = "right";
        context.fillText(gridValue.toFixed(options.precision || 4), padding.left - 8, gridY + 3);
    }

    // Bars
    for (let groupIndex = 0; groupIndex < groups.length; groupIndex++) {
        const group = groups[groupIndex];
        const groupX = padding.left + groupIndex * groupWidth + groupPadding;

        for (let barIndex = 0; barIndex < group.bars.length; barIndex++) {
            const bar = group.bars[barIndex];
            const barHeight = (bar.value / maxValue) * chartHeight;
            const x = groupX + barIndex * barWidth;
            const y = padding.top + chartHeight - barHeight;

            context.fillStyle = bar.colour || "#6366f1";
            context.globalAlpha = 0.85;
            context.fillRect(x, y, barWidth - 2, barHeight);
            context.globalAlpha = 1;
        }

        // Group label
        context.fillStyle = "#94a3b8";
        context.font = "11px Inter";
        context.textAlign = "center";
        context.fillText(group.label, groupX + (barsPerGroup * barWidth) / 2, displayHeight - 35);
    }

    // Legend
    if (groups[0] && groups[0].bars.length > 0) {
        const legendY = displayHeight - 15;
        let legendX = padding.left;
        const allLabels = new Set();
        for (const group of groups) {
            for (const bar of group.bars) {
                if (!allLabels.has(bar.label)) {
                    allLabels.add(bar.label);
                    context.fillStyle = bar.colour;
                    context.fillRect(legendX, legendY - 8, 10, 10);
                    context.fillStyle = "#94a3b8";
                    context.font = "9px Inter";
                    context.textAlign = "left";
                    context.fillText(bar.label, legendX + 14, legendY);
                    legendX += context.measureText(bar.label).width + 30;
                }
            }
        }
    }
}

// ============================================================
// RENDER TRAINING CHARTS
// ============================================================

function renderTrainingCharts() {
    if (!demoData || !demoData.training_curves) return;

    const seedColours = {
        reinforcement_learning: ["#6366f1", "#818cf8", "#a5b4fc"],
        predictive: ["#2ecc71", "#58d68d", "#82e0aa"],
    };

    // Build multi-seed reward overlay
    const rewardDatasets = [];
    const validationDatasets = [];
    const predictorDatasets = [];
    let bestPredictive = null;

    for (const [key, data] of Object.entries(demoData.training_curves)) {
        const rolling = data.rolling_rewards || [];
        const isPredictive = key.includes("predictive");
        const palette = isPredictive ? seedColours.predictive : seedColours.reinforcement_learning;
        const seedNumber = parseInt(key.match(/seed_(\d+)/)?.[1] || "0");
        const colourIndex = Math.min((seedNumber - 42) % 3, palette.length - 1);
        const typeLabel = isPredictive ? "Predictive" : "Reinforcement Learning";

        rewardDatasets.push({
            data: rolling,
            colour: palette[colourIndex >= 0 ? colourIndex : 0],
            lineWidth: isPredictive ? 2 : 1.5,
            alpha: isPredictive ? 1 : 0.6,
            label: `${typeLabel} (Seed ${seedNumber})`,
        });

        if (data.validation_rewards && data.validation_rewards.length > 0) {
            validationDatasets.push({
                data: data.validation_rewards,
                colour: palette[colourIndex >= 0 ? colourIndex : 0],
                lineWidth: 1.5,
                label: `${typeLabel} (Seed ${seedNumber})`,
            });
        }

        if (isPredictive && data.predictor_losses && data.predictor_losses.length > 0) {
            predictorDatasets.push({
                data: data.predictor_losses,
                colour: palette[colourIndex >= 0 ? colourIndex : 0],
                lineWidth: 1.5,
                label: `Seed ${seedNumber}`,
            });
            if (!bestPredictive) bestPredictive = data;
        }
    }

    if (!bestPredictive) bestPredictive = Object.values(demoData.training_curves)[0];

    drawLineChart("training-reward-chart", rewardDatasets, { xLabel: "Epoch", height: 300 });
    drawLineChart("validation-reward-chart", validationDatasets, { xLabel: "Validation Step", height: 220 });
    drawLineChart("predictor-loss-chart", predictorDatasets, { xLabel: "Training Step", height: 220 });

    // Training info cards
    const infoContainer = document.getElementById("training-info-cards");
    if (infoContainer && bestPredictive) {
        const finalEpoch = bestPredictive.final_epoch || 0;
        const stoppedEarly = bestPredictive.stopped_early || false;
        const totalTime = demoData.metadata ? demoData.metadata.total_training_time_minutes : 0;
        infoContainer.innerHTML = `
            <div class="glass-card" style="text-align:center;">
                <div class="card-icon">🎯</div>
                <h3>Final Epoch</h3>
                <p style="font-size:2rem;font-weight:800;color:var(--accent-3);">${finalEpoch.toLocaleString()}</p>
            </div>
            <div class="glass-card" style="text-align:center;">
                <div class="card-icon">🛑</div>
                <h3>Early Stopping</h3>
                <p style="font-size:2rem;font-weight:800;color:${stoppedEarly ? '#2ecc71' : '#e74c3c'};">${stoppedEarly ? 'Triggered' : 'Not triggered'}</p>
            </div>
            <div class="glass-card" style="text-align:center;">
                <div class="card-icon">⏱️</div>
                <h3>Total Training Time</h3>
                <p style="font-size:2rem;font-weight:800;color:var(--accent-3);">${totalTime} min</p>
            </div>
        `;
    }
}

// ============================================================
// RENDER RESULTS CHARTS
// ============================================================

function renderResultsCharts() {
    if (!demoData || !demoData.main_results) return;

    const stats = demoData.main_results;
    const methodColours = {
        "baseline": "#e74c3c",
        "random": "#f39c12",
        "reinforcement_learning": "#3498db",
        "predictive_reinforcement_learning": "#2ecc71",
        "predictive_reinforcement_learning_redundancy": "#9b59b6",
    };
    const methodLabels = {
        "baseline": "Baseline",
        "random": "Random",
        "reinforcement_learning": "Reinforcement Learning",
        "predictive_reinforcement_learning": "Predictive Reinforcement Learning",
        "predictive_reinforcement_learning_redundancy": "Predictive Reinforcement Learning + Redundancy",
    };

    const sequenceLengths = [...new Set(stats.map(s => s.seq_length))].sort((a,b) => a - b);
    const methods = [...new Set(stats.map(s => s.method))];

    // Error rate chart
    const errorRateGroups = sequenceLengths.map(sequenceLength => ({
        label: `Length ${sequenceLength}`,
        bars: methods.map(method => {
            const entry = stats.find(s => s.method === method && s.seq_length === sequenceLength);
            return {
                value: entry ? (entry.average_error_rate_mean || 0) : 0,
                colour: methodColours[method] || "#95a5a6",
                label: methodLabels[method] || method,
            };
        }),
    }));
    drawBarChart("error-rate-chart", errorRateGroups, { precision: 4, height: 300 });

    // Accuracy chart
    const accuracyGroups = sequenceLengths.map(sequenceLength => ({
        label: `Length ${sequenceLength}`,
        bars: methods.map(method => {
            const entry = stats.find(s => s.method === method && s.seq_length === sequenceLength);
            return {
                value: entry ? (entry.decode_accuracy_mean || 0) : 0,
                colour: methodColours[method] || "#95a5a6",
                label: methodLabels[method] || method,
            };
        }),
    }));
    drawBarChart("accuracy-chart", accuracyGroups, { precision: 1, height: 300 });
}

// ============================================================
// RENDER RESULTS TABLE
// ============================================================

function renderResultsTable() {
    if (!demoData || !demoData.main_results) return;
    const tbody = document.getElementById("results-tbody");
    if (!tbody) return;

    const methodLabels = {
        "baseline": "Fixed Baseline",
        "random": "Random Mapping",
        "reinforcement_learning": "Reinforcement Learning (Proximal Policy Optimisation)",
        "predictive_reinforcement_learning": "Predictive Reinforcement Learning",
        "predictive_reinforcement_learning_redundancy": "Predictive Reinforcement Learning + Redundancy",
    };

    tbody.innerHTML = "";
    for (const stat of demoData.main_results) {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${methodLabels[stat.method] || stat.method}</td>
            <td>${stat.seq_length}</td>
            <td>${(stat.average_error_rate_mean || 0).toFixed(4)} ± ${(stat.average_error_rate_std || 0).toFixed(4)}</td>
            <td>${(stat.decode_accuracy_mean || 0).toFixed(2)} ± ${(stat.decode_accuracy_std || 0).toFixed(2)}%</td>
            <td>${(stat.gc_score_mean || 0).toFixed(3)}</td>
        `;
        tbody.appendChild(row);
    }
}

// ============================================================
// GENERALISATION TEST
// ============================================================

function renderGeneralisationTest() {
    if (!demoData || !demoData.unseen_test_results || !demoData.main_results) return;
    const container = document.getElementById("generalisation-grid");
    if (!container) return;
    container.innerHTML = "";

    const sequenceLengths = Object.keys(demoData.unseen_test_results).sort((a, b) => parseInt(a) - parseInt(b));

    for (const lengthKey of sequenceLengths) {
        const testData = demoData.unseen_test_results[lengthKey];
        const testAccuracy = testData.decode_accuracy;

        // Find matching training accuracy (predictive reinforcement learning method)
        const trainStat = demoData.main_results.find(
            s => s.method === "predictive_reinforcement_learning" && s.seq_length === parseInt(lengthKey)
        );
        const trainAccuracy = trainStat ? trainStat.decode_accuracy_mean : testAccuracy;
        const gap = Math.abs(trainAccuracy - testAccuracy);
        const isPass = gap < 5;

        const card = document.createElement("div");
        card.className = "generalisation-card";
        card.innerHTML = `
            <div class="length-badge">Length ${lengthKey}</div>
            <div class="metric-pair">
                <div class="metric-item">
                    <div class="metric-value" style="color:var(--accent-3);">${trainAccuracy.toFixed(1)}%</div>
                    <div class="metric-label">Training Accuracy</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value" style="color:#2ecc71;">${testAccuracy.toFixed(1)}%</div>
                    <div class="metric-label">Test Accuracy (Unseen)</div>
                </div>
            </div>
            <div class="gap-indicator ${isPass ? 'pass' : 'warn'}">
                ${isPass ? '✓' : '⚠'} Gap: ${gap.toFixed(2)}%
            </div>
        `;
        container.appendChild(card);
    }
}

// ============================================================
// REDUNDANCY ANALYSIS CHART
// ============================================================

function renderRedundancyAnalysisChart() {
    if (!demoData || !demoData.redundancy_comparison) return;

    const comparison = demoData.redundancy_comparison;
    const sequenceLengths = Object.keys(comparison).sort((a, b) => parseInt(a) - parseInt(b));
    const copyColours = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6"];

    // Build grouped bar data: groups = copy counts, bars = sequence lengths
    const copyCountsToShow = [1, 2, 3, 4, 5];
    const groups = copyCountsToShow.map((copyCount, index) => ({
        label: `${copyCount} ${copyCount === 1 ? 'Copy' : 'Copies'}`,
        bars: sequenceLengths.map(lengthKey => {
            const entry = comparison[lengthKey].find(c => c.number_of_copies === copyCount);
            return {
                value: entry ? entry.redundancy_accuracy * 100 : 0,
                colour: {
                    "64": "#6366f1",
                    "128": "#06b6d4",
                    "256": "#f59e0b",
                }[lengthKey] || "#95a5a6",
                label: `Length ${lengthKey}`,
            };
        }),
    }));

    drawBarChart("redundancy-analysis-chart", groups, { precision: 1, height: 300 });
}

// ============================================================
// ABLATION CHART
// ============================================================

function renderAblationChart() {
    if (!demoData || !demoData.ablation_results) return;

    const stats = demoData.ablation_results;
    const methodColours = {
        "baseline": "#e74c3c",
        "reinforcement_learning": "#3498db",
        "predictive_reinforcement_learning": "#2ecc71",
        "reinforcement_learning_no_gc": "#f39c12",
        "reinforcement_learning_no_homopolymer": "#9b59b6",
    };
    const methodLabels = {
        "baseline": "Fixed Baseline",
        "reinforcement_learning": "Full Model",
        "predictive_reinforcement_learning": "Predictive Reinforcement Learning",
        "reinforcement_learning_no_gc": "Without GC Reward",
        "reinforcement_learning_no_homopolymer": "Without Homopolymer Penalty",
    };

    const sequenceLengths = [...new Set(stats.map(s => s.seq_length))].sort((a, b) => a - b);
    const methods = [...new Set(stats.map(s => s.method))];

    const groups = sequenceLengths.map(sequenceLength => ({
        label: `Length ${sequenceLength}`,
        bars: methods.map(method => {
            const entry = stats.find(s => s.method === method && s.seq_length === sequenceLength);
            return {
                value: entry ? (entry.average_error_rate_mean || 0) : 0,
                colour: methodColours[method] || "#95a5a6",
                label: methodLabels[method] || method,
            };
        }),
    }));

    drawBarChart("ablation-chart", groups, { precision: 4, height: 300 });
}

// ============================================================
// ERROR LABORATORY
// ============================================================

function initializeErrorLaboratory() {
    const sliderSubstitution = document.getElementById("slider-substitution");
    const sliderInsertion = document.getElementById("slider-insertion");
    const sliderDeletion = document.getElementById("slider-deletion");

    if (!sliderSubstitution) return;

    // Generate a sample DNA strand
    const sampleDNA = Array.from({ length: 80 }, () =>
        NUCLEOTIDES[Math.floor(Math.random() * 4)]
    ).join("");

    function updateLabSimulation() {
        const substitutionRate = parseFloat(sliderSubstitution.value) / 100;
        const insertionRate = parseFloat(sliderInsertion.value) / 100;
        const deletionRate = parseFloat(sliderDeletion.value) / 100;

        document.getElementById("val-substitution").textContent = (substitutionRate * 100).toFixed(1) + "%";
        document.getElementById("val-insertion").textContent = (insertionRate * 100).toFixed(1) + "%";
        document.getElementById("val-deletion").textContent = (deletionRate * 100).toFixed(1) + "%";

        renderDNAStrand(document.getElementById("lab-original-dna"), sampleDNA);

        const { mutatedSequence, statistics, errorDetails } = simulateErrors(
            sampleDNA, substitutionRate, insertionRate, deletionRate
        );
        renderDNAStrand(document.getElementById("lab-mutated-dna"), mutatedSequence, errorDetails);

        document.getElementById("lab-stats").innerHTML = `
            <span class="lab-stat" style="color:var(--error-substitution);"><strong>Substitutions:</strong> ${statistics.substitutions}</span>
            <span class="lab-stat" style="color:var(--error-insertion);"><strong>Insertions:</strong> ${statistics.insertions}</span>
            <span class="lab-stat" style="color:var(--error-deletion);"><strong>Deletions:</strong> ${statistics.deletions}</span>
            <span class="lab-stat"><strong>Original length:</strong> ${sampleDNA.length}</span>
            <span class="lab-stat"><strong>Mutated length:</strong> ${mutatedSequence.length}</span>
        `;

        // Redundancy comparison chart
        const accuracies = [];
        const copyCountsToTest = [1, 2, 3, 4, 5];
        for (const count of copyCountsToTest) {
            const copies = [];
            for (let copyIndex = 0; copyIndex < count; copyIndex++) {
                const result = simulateErrors(sampleDNA, substitutionRate, insertionRate, deletionRate);
                copies.push(result.mutatedSequence);
            }
            const consensusResult = majorityVote(copies);
            let matchCount = 0;
            const compareLength = Math.min(sampleDNA.length, consensusResult.length);
            for (let position = 0; position < compareLength; position++) {
                if (sampleDNA[position] === consensusResult[position]) matchCount++;
            }
            accuracies.push(matchCount / sampleDNA.length * 100);
        }

        drawLineChart("redundancy-chart", [{
            data: accuracies,
            colour: "#8b5cf6",
            lineWidth: 3,
        }], { xLabel: "Number of Copies", height: 250 });
    }

    sliderSubstitution.addEventListener("input", updateLabSimulation);
    sliderInsertion.addEventListener("input", updateLabSimulation);
    sliderDeletion.addEventListener("input", updateLabSimulation);

    updateLabSimulation();
}

// ============================================================
// DNA HELIX BACKGROUND ANIMATION
// ============================================================

function initializeHelixAnimation() {
    const container = document.getElementById("dna-canvas");
    if (!container) return;

    const canvas = document.createElement("canvas");
    container.appendChild(canvas);
    canvas.style.cssText = "width:100%;height:100%;";
    const context = canvas.getContext("2d");

    function resize() {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
    }
    resize();
    window.addEventListener("resize", resize);

    let time = 0;
    function animate() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        time += 0.008;

        const centerX = canvas.width / 2;
        const amplitude = Math.min(canvas.width * 0.15, 150);
        const verticalSpacing = 25;
        const numberOfRungs = Math.ceil(canvas.height / verticalSpacing) + 4;

        for (let rung = -2; rung < numberOfRungs; rung++) {
            const y = rung * verticalSpacing;
            const phase = time + rung * 0.3;
            const offset1 = Math.sin(phase) * amplitude;
            const offset2 = Math.sin(phase + Math.PI) * amplitude;

            const depth1 = (Math.cos(phase) + 1) / 2;
            const depth2 = (Math.cos(phase + Math.PI) + 1) / 2;

            // Rungs
            context.strokeStyle = `rgba(99,102,241,${0.05 + depth1 * 0.08})`;
            context.lineWidth = 1;
            context.beginPath();
            context.moveTo(centerX + offset1, y);
            context.lineTo(centerX + offset2, y);
            context.stroke();

            // Strand 1 node
            context.fillStyle = `rgba(99,102,241,${0.1 + depth1 * 0.2})`;
            context.beginPath();
            context.arc(centerX + offset1, y, 2 + depth1 * 2, 0, Math.PI * 2);
            context.fill();

            // Strand 2 node
            context.fillStyle = `rgba(139,92,246,${0.1 + depth2 * 0.2})`;
            context.beginPath();
            context.arc(centerX + offset2, y, 2 + depth2 * 2, 0, Math.PI * 2);
            context.fill();
        }

        requestAnimationFrame(animate);
    }
    animate();
}

// ============================================================
// NAVBAR
// ============================================================

function initializeNavbar() {
    const navbar = document.getElementById("navbar");
    const toggle = document.querySelector(".nav-toggle");
    const links = document.querySelector(".nav-links");

    if (toggle && links) {
        toggle.addEventListener("click", () => {
            links.style.display = links.style.display === "flex" ? "none" : "flex";
        });
    }

    // Active link highlighting
    const sections = document.querySelectorAll("section[id]");
    window.addEventListener("scroll", () => {
        const scrollPosition = window.scrollY + 100;
        for (const section of sections) {
            if (scrollPosition >= section.offsetTop && scrollPosition < section.offsetTop + section.offsetHeight) {
                document.querySelectorAll(".nav-links a").forEach(link => {
                    link.classList.remove("active");
                    if (link.getAttribute("href") === "#" + section.id) {
                        link.classList.add("active");
                    }
                });
            }
        }
    });
}

// ============================================================
// INITIALISATION
// ============================================================

document.addEventListener("DOMContentLoaded", function() {
    initializeHelixAnimation();
    initializeNavbar();
    renderMappingsGrid();
    initializeErrorLaboratory();
    loadDemoData();

    // Pipeline button
    const runButton = document.getElementById("run-pipeline-button");
    if (runButton) {
        runButton.addEventListener("click", runPipeline);
    }
});
