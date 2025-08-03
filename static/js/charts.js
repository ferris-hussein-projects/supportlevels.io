/**
 * Chart.js utilities and configurations for Stock Support Tracker
 */

// Chart.js default configuration for dark theme
Chart.defaults.color = '#ffffff';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
Chart.defaults.backgroundColor = 'rgba(255, 255, 255, 0.1)';

// Common chart options
const commonChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
        intersect: false,
        mode: 'index'
    },
    scales: {
        y: {
            beginAtZero: false,
            grid: {
                color: 'rgba(255, 255, 255, 0.1)'
            },
            ticks: {
                color: '#ffffff'
            }
        },
        x: {
            grid: {
                color: 'rgba(255, 255, 255, 0.1)'
            },
            ticks: {
                color: '#ffffff'
            }
        }
    },
    plugins: {
        legend: {
            display: true,
            position: 'top',
            labels: {
                color: '#ffffff'
            }
        },
        tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: '#ffffff',
            bodyColor: '#ffffff',
            borderColor: 'rgba(255, 255, 255, 0.2)',
            borderWidth: 1
        }
    }
};

// Color palette for charts
const chartColors = {
    primary: 'rgb(75, 192, 192)',
    secondary: 'rgb(255, 159, 64)',
    tertiary: 'rgb(153, 102, 255)',
    success: 'rgb(75, 192, 75)',
    danger: 'rgb(255, 99, 132)',
    warning: 'rgb(255, 205, 86)',
    info: 'rgb(54, 162, 235)'
};

/**
 * Create a price chart with moving averages
 * @param {string} canvasId - ID of the canvas element
 * @param {Object} data - Chart data from API
 * @returns {Chart} Chart.js instance
 */
function createPriceChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Price',
                    data: data.prices,
                    borderColor: chartColors.primary,
                    backgroundColor: chartColors.primary.replace('rgb', 'rgba').replace(')', ', 0.1)'),
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 4
                },
                {
                    label: '21-day MA',
                    data: data.ma21,
                    borderColor: chartColors.secondary,
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    pointRadius: 0,
                    pointHoverRadius: 3
                },
                {
                    label: '50-day MA',
                    data: data.ma50,
                    borderColor: chartColors.tertiary,
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    pointRadius: 0,
                    pointHoverRadius: 3
                }
            ]
        },
        options: {
            ...commonChartOptions,
            scales: {
                ...commonChartOptions.scales,
                y: {
                    ...commonChartOptions.scales.y,
                    title: {
                        display: true,
                        text: 'Price ($)',
                        color: '#ffffff'
                    }
                },
                x: {
                    ...commonChartOptions.scales.x,
                    title: {
                        display: true,
                        text: 'Date',
                        color: '#ffffff'
                    }
                }
            }
        }
    });
}

/**
 * Create a volume chart
 * @param {string} canvasId - ID of the canvas element
 * @param {Object} data - Chart data from API
 * @returns {Chart} Chart.js instance
 */
function createVolumeChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Volume',
                    data: data.volumes,
                    backgroundColor: chartColors.info.replace('rgb', 'rgba').replace(')', ', 0.6)'),
                    borderColor: chartColors.info,
                    borderWidth: 1
                }
            ]
        },
        options: {
            ...commonChartOptions,
            scales: {
                ...commonChartOptions.scales,
                y: {
                    ...commonChartOptions.scales.y,
                    title: {
                        display: true,
                        text: 'Volume',
                        color: '#ffffff'
                    }
                },
                x: {
                    ...commonChartOptions.scales.x,
                    title: {
                        display: true,
                        text: 'Date',
                        color: '#ffffff'
                    }
                }
            }
        }
    });
}

/**
 * Create a comparison chart for multiple stocks
 * @param {string} canvasId - ID of the canvas element
 * @param {Array} stocksData - Array of stock data objects
 * @returns {Chart} Chart.js instance
 */
function createComparisonChart(canvasId, stocksData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const datasets = stocksData.map((stock, index) => ({
        label: stock.ticker,
        data: stock.prices,
        borderColor: Object.values(chartColors)[index % Object.values(chartColors).length],
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.1
    }));
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: stocksData[0].dates, // Assume all stocks have same date range
            datasets: datasets
        },
        options: {
            ...commonChartOptions,
            scales: {
                ...commonChartOptions.scales,
                y: {
                    ...commonChartOptions.scales.y,
                    title: {
                        display: true,
                        text: 'Normalized Price (%)',
                        color: '#ffffff'
                    }
                }
            }
        }
    });
}

/**
 * Create a pie chart for sector breakdown
 * @param {string} canvasId - ID of the canvas element
 * @param {Object} sectorData - Sector breakdown data
 * @returns {Chart} Chart.js instance
 */
function createSectorChart(canvasId, sectorData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    const labels = Object.keys(sectorData);
    const data = Object.values(sectorData);
    const backgroundColors = labels.map((_, index) => 
        Object.values(chartColors)[index % Object.values(chartColors).length]
    );
    
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColors,
                borderColor: '#ffffff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'right',
                    labels: {
                        color: '#ffffff',
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ffffff',
                    bodyColor: '#ffffff',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed * 100) / total).toFixed(1);
                            return `${context.label}: ${context.parsed} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Format large numbers for display
 * @param {number} num - Number to format
 * @returns {string} Formatted number string
 */
function formatNumber(num) {
    if (num >= 1e9) {
        return (num / 1e9).toFixed(1) + 'B';
    } else if (num >= 1e6) {
        return (num / 1e6).toFixed(1) + 'M';
    } else if (num >= 1e3) {
        return (num / 1e3).toFixed(1) + 'K';
    }
    return num.toLocaleString();
}

/**
 * Show loading spinner on chart canvas
 * @param {string} canvasId - ID of the canvas element
 */
function showChartLoading(canvasId) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#ffffff';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Loading...', canvas.width / 2, canvas.height / 2);
}

/**
 * Show error message on chart canvas
 * @param {string} canvasId - ID of the canvas element
 * @param {string} message - Error message to display
 */
function showChartError(canvasId, message = 'Error loading chart') {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#ff6b6b';
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

// Export functions for use in other scripts
window.StockCharts = {
    createPriceChart,
    createVolumeChart,
    createComparisonChart,
    createSectorChart,
    formatNumber,
    showChartLoading,
    showChartError,
    chartColors,
    commonChartOptions
};
