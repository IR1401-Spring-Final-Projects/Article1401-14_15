let params = {
    elastic: true,
    type: 'tf-idf',
    by: 'title',
    expression: ''
};

function onSadeghClicked() {
    const engine_value = document.getElementById('engine').value;
    if (engine_value === 'hide') {
        alert('Please select an engine.');
        return;
    } else {
        params.elastic = (engine_value === 'elasticsearch');
    }

    const type_value = document.getElementById('type').value;
    if (type_value === 'hide') {
        alert('Please select a type.');
        return;
    } else {
        params.elastic = type_value;
    }

    const attribute_value = document.getElementById('attribute').value;
    if (attribute_value === 'hide') {
        alert('Please select an attribute.');
        return;
    } else {
        params.elastic = attribute_value;
    }

    window.location.href = './search.html';
}

function onSearchClick() {
    const text = document.getElementById('searchField').value;
    params.expression = text;
    let getParams = '';

    for (var key of Object.keys(params)) {
        getParams = getParams + (key + "=" + params[key] + '&');
    }
    getParams = getParams.slice(0, -1);
    const url = 'http://localhost:8000/search/sadegh/?' + getParams;

    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            showResponseData(xmlHttp.responseText);
    }
    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);

    document.getElementById('searchField').value = '';
}

function showResponseData(response) {
    console.log(response);
}
