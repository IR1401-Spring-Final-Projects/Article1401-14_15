function onSadeghClicked() {
    let params = {};
    const engine_value = document.getElementById('engine').value;
    if (engine_value === 'hide') {
        alert('Please select an engine.');
        return;
    } else {
        params.elastic = (engine_value === 'elasticsearch') ? 1 : 0;
    }

    const type_value = document.getElementById('type').value;
    if (type_value === 'hide') {
        alert('Please select a type.');
        return;
    } else {
        params.type = type_value;
    }

    const attribute_value = document.getElementById('attribute').value;
    if (attribute_value === 'hide') {
        alert('Please select an attribute.');
        return;
    } else {
        params.by = attribute_value;
    }

    localStorage.setItem('params', JSON.stringify(params));

    window.location.href = './search.html';
}

function onSearchClick() {
    let retrievedObject = localStorage.getItem('params');
    params = JSON.parse(retrievedObject);
    console.log('params: ', params);
    const text = document.getElementById('searchField').value;
    const expandQuery = document.getElementById('query-expansion-checkbox').checked ? 1 : 0;
    params.expression = text;
    params.expand = expandQuery;
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

function onClassificationClicked() {
    const element = document.getElementById('class-cluster-txt');
    const text = element.value;

    var data = new FormData();
    data.append('text', text);
    const url = 'http://localhost:8000/search/classify/';

    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
            resp = JSON.parse(xmlHttp.responseText)
            document.getElementById('cls-result').innerHTML = `class: ${resp.class}`;
        }
    }
    xmlHttp.open("POST", url, true);
    xmlHttp.send(data);

    element.value = '';
}

function onClusteringClicked() {
    const element = document.getElementById('class-cluster-txt');
    const text = element.value;

    var data = new FormData();
    data.append('text', text);
    const url = 'http://localhost:8000/search/cluster/';

    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200) {
            resp = JSON.parse(xmlHttp.responseText)
            document.getElementById('cls-result').innerHTML = `cluster: ${resp.cluster}`;
        }
    }
    xmlHttp.open("POST", url, true);
    xmlHttp.send(data);

    element.value = '';
}

function showResponseData(response) {
    console.log(response);

    resp = JSON.parse(response);

    table = document.getElementById('search-result');

    for (let result of resp) {
        var tr = document.createElement('tr');

        var td1 = document.createElement('td');
        var td2 = document.createElement('td');
        var td3 = document.createElement('td');

        var text1 = document.createTextNode(result.id);
        var text2 = document.createTextNode(result.scorce ? result.scorce : '');
        var text3 = document.createTextNode(result.title);

        td1.appendChild(text1);
        td2.appendChild(text2);
        td3.appendChild(text3);
        tr.appendChild(td1);
        tr.appendChild(td2);
        tr.appendChild(td3);

        table.appendChild(tr);
    }

}
