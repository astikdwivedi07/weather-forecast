// Wait for DOM to load
document.addEventListener('DOMContentLoaded', () => {
    // ================================
    // 📊 TEMPERATURE CHART SETUP
    // ================================
    const chartElement = document.getElementById('chart');
    if (!chartElement) {
        console.error('Canvas Element not found.');
        return;
    }

    const ctx = chartElement.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 150);
    gradient.addColorStop(0, 'rgba(255, 99, 132, 1)');
    gradient.addColorStop(1, 'rgba(54, 162, 235, 0.6)');

    const forecastItems = document.querySelectorAll('.forecast-item');
    const temps = [];
    const times = [];

    forecastItems.forEach(item => {
        const time = item.querySelector('.forecast-time')?.textContent?.trim();
        const temp = item.querySelector('.forecast-temperatureValue')?.textContent?.trim();

        if (time && temp) {
            times.push(time);
            temps.push(parseFloat(temp));
        }
    });

    if (temps.length === 0 || times.length === 0) {
        console.warn('Temperature or time values missing.');
        return;
    }

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [{
                label: 'Temperature (°C)',
                data: temps,
                borderColor: gradient,
                borderWidth: 3,
                tension: 0.4,
                pointRadius: 4,
                pointBackgroundColor: '#ffffff',
                pointBorderWidth: 2,
                fill: false,
            }],
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    enabled: true,
                    backgroundColor: 'rgba(0,0,0,0.6)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    bodyFont: { weight: 'bold' },
                },
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        drawOnChartArea: false,
                        color: 'rgba(255,255,255,0.1)',
                    },
                    ticks: {
                        color: '#fff',
                        font: { size: 12 }
                    },
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(255,255,255,0.1)',
                    },
                    ticks: {
                        color: '#fff',
                        font: { size: 12 }
                    },
                },
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuad',
            },
        },
    });

    // ================================
    // 🎬 DYNAMIC BACKGROUND VIDEO
    // ================================
    const weatherMain = document.querySelector('main');
    const videoElement = document.getElementById('weatherVideo');

    if (!weatherMain || !videoElement) {
        console.warn('Weather or video element missing.');
        return;
    }

    const desc = weatherMain.className.toLowerCase();
    let videoSrc = "{% static 'videos/default.mp4' %}";

    if (desc.includes('rain')) videoSrc = "{% static 'videos/rain.mp4' %}";
    else if (desc.includes('snow')) videoSrc = "{% static 'videos/snow.mp4' %}";
    else if (desc.includes('cloud')) videoSrc = "{% static 'videos/clouds.mp4' %}";
    else if (desc.includes('clear') || desc.includes('sun')) videoSrc = "{% static 'videos/sunny.mp4' %}";
    else if (desc.includes('night')) videoSrc = "{% static 'videos/night.mp4' %}";
    else if (desc.includes('fog') || desc.includes('mist') || desc.includes('haze')) videoSrc = "{% static 'videos/fog.mp4' %}";
    else if (desc.includes('thunder')) videoSrc = "{% static 'videos/thunder.mp4' %}";

    // Smooth transition effect
    videoElement.style.opacity = '0';
    setTimeout(() => {
        videoElement.querySelector('source').setAttribute('src', videoSrc);
        videoElement.load();
        videoElement.play().catch(err => console.warn('Autoplay blocked:', err));
        videoElement.style.opacity = '1';
    }, 300);
});
