$('.result-error').each(function() {
    var elem = $(this),
        timer = 0,
        interval = 200,
        stopAfter = 3000;
    refreshIntervalId = setInterval(function() {
        if (elem.css('visibility') == 'hidden') {
            elem.css('visibility', 'visible');
            if(timer > stopAfter) {
                clearInterval(refreshIntervalId);
            }
        } else {
            elem.css('visibility', 'hidden');
        }
        timer += interval;
    }, interval);
});

$('.result-success').each(function() {
    var elem = $(this),
        timer = 0,
        interval = 500,
        stopAfter = 3000;
    refreshIntervalId = setInterval(function() {
        if (elem.css('visibility') == 'hidden') {
            elem.css('visibility', 'visible');
            if(timer > stopAfter) {
                clearInterval(refreshIntervalId);
            }
        } else {
            elem.css('visibility', 'hidden');
        }
        timer += interval;
    }, interval);
});