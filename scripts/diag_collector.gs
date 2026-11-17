/**
 * Diagnostics collector for the Tami-Nautica GitHub Pages site.
 *
 * Setup (once):
 *  1. Create a Google Sheet, open Extensions > Apps Script, paste this file.
 *  2. Deploy > New deployment > type "Web app".
 *       Execute as: Me.  Who has access: Anyone.
 *  3. Copy the web-app URL (ends in /exec) into DIAG_REMOTE_URL in
 *     prototype/index.html, then redeploy the site (scripts/deploy_pages.sh --push).
 *
 * Events arrive as text/plain JSON via navigator.sendBeacon. Each becomes a row.
 * No cookies and no personal data: random per-visit session id, UA, viewport, errors.
 */
var MAX_ROWS = 20000;

function doPost(e) {
  try {
    var d = JSON.parse(e.postData.contents);
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
    if (sheet.getLastRow() === 0) {
      sheet.appendRow(['time', 'sid', 't_ms', 'event', 'ua', 'detail']);
    }
    if (sheet.getLastRow() >= MAX_ROWS) return ContentService.createTextOutput('full');
    var detail = d.detail || {};
    sheet.appendRow([
      new Date(),
      String(d.sid || '').slice(0, 16),
      Number(d.t_ms) || 0,
      String(d.event || '').slice(0, 64),
      String(detail.ua || '').slice(0, 300),
      JSON.stringify(detail).slice(0, 2000)
    ]);
  } catch (err) {}
  return ContentService.createTextOutput('ok');
}
