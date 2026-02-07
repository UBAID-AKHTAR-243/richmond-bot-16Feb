(function() {
    // 1. Load FontAwesome immediately
    if (!document.getElementById('fa-styles')) {
        const faLink = document.createElement('link');
        faLink.id = 'fa-styles';
        faLink.rel = 'stylesheet';
        faLink.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css';
        document.head.appendChild(faLink);
    }

    // 2. Optimized & Beautified CSS
    const style = document.createElement('style');
    style.innerHTML = `
        #bot-widget-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 99999;
            font-family: -apple-system, BlinkMacSystemFont, Segue, Roboto, Helvetica, Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            transition: all 0.3s ease;
        }

        /* Responsive behavior for mobile */
        @media (max-width: 480px) {
            #bot-widget-container {
                bottom: 15px;
                right: 15px;
            }
        }

        #bot-window-frame {
            display: none;
            width: 360px;
            height: 540px;
            max-height: 80vh;
            background: #fff; /* Light mode default */
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            margin-bottom: 15px;
            overflow: hidden;
            border: 1px solid #ccc;
            transition: all 0.3s ease;
        }

        #bot-window-frame.active {
            display: flex;
            flex-direction: column;
            animation: slideUp 0.3s ease-out;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px) scale(0.95); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }

        /* WhatsApp FAB Button - Improved */
        .whatsapp-btn {
            position: relative;
            width: 60px;
            height: 60px;
            background-color: #25d366;
            color: white;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(37, 211, 102, 0.4);
            transition: all 0.2s ease-in-out;
            outline: none;
            -webkit-tap-highlight-color: transparent;
        }

        .whatsapp-btn:hover {
            background-color: #128c7e;
            transform: scale(1.05);
            box-shadow: 0 6px 16px rgba(18, 140, 126, 0.5);
        }

        .whatsapp-btn:active {
            transform: scale(0.95);
        }

        .whatsapp-btn i {
            font-size: 32px;
        }

        /* Pulse Animation */
        .whatsapp-btn::after {
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background-color: #25d366;
            z-index: -1;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% {
                transform: scale(1);
                opacity: 0.6;
            }
            100% {
                transform: scale(1.5);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);

    // 3. Create the HTML Wrapper
    const widget = document.createElement('div');
    widget.id = 'bot-widget-container';
    widget.innerHTML = `
        <div id="bot-window-frame">
            <div id="bot-content-placeholder" style="height: 100%; width: 100%;"></div>
        </div>
        <button class="whatsapp-btn" id="toggle-bot" title="Chat with us">
            <i class="fab fa-whatsapp"></i>
        </button>
    `;
    document.body.appendChild(widget);

    const btn = document.getElementById('toggle-bot');
    const windowFrame = document.getElementById('bot-window-frame');
    const placeholder = document.getElementById('bot-content-placeholder');

    btn.onclick = function() {
        const isActive = windowFrame.classList.toggle('active');
        
        // Change icon to close when open, back to WhatsApp when closed
        const icon = btn.querySelector('i');
        if (isActive) {
            icon.classList.remove('fa-whatsapp');
            icon.classList.add('fa-times');
            icon.style.fontSize = '24px'; // Slightly smaller for X
        } else {
            icon.classList.remove('fa-times');
            icon.classList.add('fa-whatsapp');
            icon.style.fontSize = '32px';
        }

        if (isActive && placeholder.innerHTML === "") {
            loadBotContent();
        }
    };

    function loadBotContent() {
        // Placeholder text or loading spinner
        placeholder.innerHTML = '<div style="padding:20px; color:#555;">Loading...</div>';
        
        fetch('bot.html')
            .then(response => response.text())
            .then(data => {
                placeholder.innerHTML = data;
                // Execute scripts if any
                const scripts = placeholder.getElementsByTagName('script');
                for (let script of scripts) {
                    const newScript = document.createElement("script");
                    if (script.src) {
                        newScript.src = script.src;
                    } else {
                        newScript.textContent = script.textContent;
                    }
                    document.body.appendChild(newScript);
                }
            })
            .catch(err => {
                placeholder.innerHTML = '<div style="padding:20px; color:red;">Error loading content.</div>';
                console.error(err);
            });
    }
})();
