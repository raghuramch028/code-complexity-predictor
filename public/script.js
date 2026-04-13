document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const codeInput = document.getElementById('codeInput');
    const resultSection = document.getElementById('resultSection');
    const loading = document.getElementById('loading');
    const output = document.getElementById('output');

    analyzeBtn.addEventListener('click', async () => {
        const code = codeInput.value.trim();
        
        if (!code) {
            alert("Please paste some code to analyze!");
            return;
        }

        // Show loading state
        resultSection.style.display = 'block';
        loading.style.display = 'flex';
        output.style.display = 'none';
        output.innerHTML = '';
        
        // Scroll to result
        resultSection.scrollIntoView({ behavior: 'smooth' });

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ code })
            });

            const data = await response.json();

            if (!response.ok) {
                // If it's an API key error, typically Google returns specific errors, but our backend might return 500
                throw new Error(data.error || 'Network response was not ok');
            }

            
            // Hide loading and show result
            loading.style.display = 'none';
            output.style.display = 'block';
            
            // Format output (Basic formatting, bolding Big-O)
            let formattedText = data.result
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>');
            
            output.innerHTML = formattedText;
            
        } catch (error) {
            console.error('Error:', error);
            loading.style.display = 'none';
            output.style.display = 'block';
            output.innerHTML = `<span style="color: #ff5f56;">Error: ${error.message}. Please check your connection and API Key if you haven't set it yet.</span>`;
        }
    });

    // Optional: Add tab key support in textarea
    codeInput.addEventListener('keydown', function(e) {
        if (e.key == 'Tab') {
            e.preventDefault();
            var start = this.selectionStart;
            var end = this.selectionEnd;

            // set textarea value to: text before caret + tab + text after caret
            this.value = this.value.substring(0, start) +
            "    " + this.value.substring(end);

            // put caret at right position again
            this.selectionStart =
            this.selectionEnd = start + 4;
        }
    });
});
