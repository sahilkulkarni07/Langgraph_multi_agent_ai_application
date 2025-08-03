async function submitQuestion() {
    const question = document.getElementById("question").value;

    const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question })  // <--- IMPORTANT: key must be 'question'
    });

    const data = await response.json();
    document.getElementById("responseArea").innerText = data.results[0];
}

