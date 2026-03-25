document.addEventListener('DOMContentLoaded', () => {
    const codeInput = document.getElementById('codeInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const btnText = analyzeBtn.querySelector('.btn-text');
    const loader = analyzeBtn.querySelector('.loader');
    
    // Result elements
    const resNotation = document.getElementById('resNotation');
    const resTitle = document.getElementById('resTitle');
    const resDesc = document.getElementById('resDesc');
    
    // Example chips
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            // Replace \n from dataset with actual newlines
            codeInput.value = chip.dataset.example.replace(/\\n/g, '\n');
            triggerAnalysis();
        });
    });

    analyzeBtn.addEventListener('click', triggerAnalysis);

    // Also support Cmd/Ctrl + Enter to trigger
    codeInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            triggerAnalysis();
        }
    });

    async function triggerAnalysis() {
        const code = codeInput.value.trim();
        
        if (!code) {
            resNotation.innerHTML = "O(?)";
            resTitle.innerHTML = "Error";
            resDesc.innerHTML = "⚠️ Please paste some code before analyzing!";
            return;
        }

        // UI Loading state
        analyzeBtn.disabled = true;
        btnText.innerHTML = "Analyzing structure...";
        loader.classList.remove('hidden');
        resNotation.style.opacity = '0.5';
        resTitle.style.opacity = '0.5';
        resDesc.style.opacity = '0.5';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ code: code })
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            
            // Render Results
            resNotation.innerHTML = data.notation;
            resTitle.innerHTML = data.title;
            resDesc.innerHTML = data.description;
            
        } catch (error) {
            console.error('Analysis failed:', error);
            resNotation.innerHTML = "O(?)";
            resTitle.innerHTML = "Analysis Failed";
            resDesc.innerHTML = "An error occurred while connecting to the AI model. Ensure the backend is running.";
        } finally {
            // Restore UI state
            analyzeBtn.disabled = false;
            btnText.innerHTML = "⚡ Analyze Complexity";
            loader.classList.add('hidden');
            
            // Fade results back in
            resNotation.style.opacity = '1';
            resTitle.style.opacity = '1';
            resDesc.style.opacity = '1';
            
            // Add a little pop animation to the results
            document.querySelectorAll('.result-card').forEach(card => {
                card.style.transform = 'scale(1.02)';
                setTimeout(() => card.style.transform = 'scale(1)', 200);
            });
        }
    }
});
