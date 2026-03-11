const API_URL = "http://localhost:5000/predict";
const API_KEY = "solar-yield-secret-2026";

// DOM Elements
const sliders = {
    irr: document.getElementById('irr'),
    amb: document.getElementById('amb_temp'),
    sys: document.getElementById('system'),
    load: document.getElementById('load'),
    soc: document.getElementById('soc')
};

const labels = {
    irr: document.getElementById('val_irr'),
    amb: document.getElementById('val_amb'),
    sys: document.getElementById('val_sys'),
    load: document.getElementById('val_load'),
    soc: document.getElementById('val_soc')
};

const results = {
    solar: document.getElementById('res_solar'),
    net: document.getElementById('res_net'),
    net_card: document.getElementById('card_net'),
    net_icon: document.getElementById('icon_net'),
    batt_energy: document.getElementById('res_batt_energy'),
    bar_soc: document.getElementById('bar_soc'),
    status_badge: document.getElementById('status_badge'),
    time: document.getElementById('res_time')
};

// Debounce function to prevent too many API calls
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Update UI Numbers from Sliders
function updateLabels() {
    labels.irr.textContent = sliders.irr.value;
    labels.amb.textContent = sliders.amb.value;
    labels.sys.textContent = sliders.sys.value;
    labels.load.textContent = sliders.load.value;
    labels.soc.textContent = sliders.soc.value;
}

// Main Prediction Function
async function updatePrediction() {
    updateLabels();

    const payload = {
        irradiation: sliders.irr.value,
        ambient_temp: sliders.amb.value,
        system_capacity: sliders.sys.value,
        load: sliders.load.value,
        soc: sliders.soc.value
    };

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'X-API-KEY': API_KEY
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const data = await response.json();
        renderResults(data);

    } catch (error) {
        console.error('Error fetching prediction:', error);
        results.time.textContent = "Error connecting to server.";
    }
}

function renderResults(data) {
    // 1. Solar
    results.solar.textContent = `${data.solar_kw.toFixed(2)} kW`;

    // 2. Net Power
    const net = data.net_power;
    const sign = net > 0 ? '+' : '';
    results.net.textContent = `${sign}${net.toFixed(2)} kW`;

    // Styling for Net Power Card
    if (net > 0.01) {
        // Charging (Green-ish)
        results.net_card.className = "bg-green-50 p-6 rounded-xl shadow-md border border-green-100 flex flex-col items-center transition-colors duration-300";
        results.net_icon.className = "p-3 bg-green-100 rounded-full mb-3 text-green-600";
    } else if (net < -0.01) {
        // Draining (Red-ish)
        results.net_card.className = "bg-red-50 p-6 rounded-xl shadow-md border border-red-100 flex flex-col items-center transition-colors duration-300";
        results.net_icon.className = "p-3 bg-red-100 rounded-full mb-3 text-red-600";
    } else {
        // Balanced
        results.net_card.className = "bg-white p-6 rounded-xl shadow-md border border-gray-100 flex flex-col items-center transition-colors duration-300";
        results.net_icon.className = "p-3 bg-gray-100 rounded-full mb-3 text-gray-600";
    }

    // 3. Battery
    results.batt_energy.textContent = `${data.battery_kwh.toFixed(2)} / ${data.battery_total_kwh.toFixed(2)} kWh`;
    
    // Bar Width
    const socPercent = (data.battery_kwh / data.battery_total_kwh) * 100;
    results.bar_soc.style.width = `${socPercent}%`;
    
    // Bar Color based on SOC
    if (socPercent < 20) results.bar_soc.className = "bg-red-500 h-6 transition-all duration-500 relative";
    else if (socPercent < 50) results.bar_soc.className = "bg-yellow-400 h-6 transition-all duration-500 relative";
    else results.bar_soc.className = "bg-green-500 h-6 transition-all duration-500 relative";

    // Status Badge & Time
    results.time.textContent = data.time_msg;
    
    if (data.status === 'charging') {
        results.status_badge.textContent = "⚡ Charging";
        results.status_badge.className = "px-4 py-2 rounded-lg text-sm font-bold bg-green-100 text-green-700 animate-pulse";
    } else if (data.status === 'draining') {
        results.status_badge.textContent = "⚠️ Draining";
        results.status_badge.className = "px-4 py-2 rounded-lg text-sm font-bold bg-red-100 text-red-700";
    } else if (data.status === 'full') {
        results.status_badge.textContent = "🔋 Full";
        results.status_badge.className = "px-4 py-2 rounded-lg text-sm font-bold bg-indigo-100 text-indigo-700";
    } else {
        results.status_badge.textContent = "⚖️ Balanced";
        results.status_badge.className = "px-4 py-2 rounded-lg text-sm font-bold bg-gray-100 text-gray-600";
    }
}

// Attack Listeners
const debouncedUpdate = debounce(updatePrediction, 50);

Object.values(sliders).forEach(slider => {
    slider.addEventListener('input', debouncedUpdate);
});

// Initial Call
updatePrediction();
