import UIKit
import WebKit
import Capacitor

/// Bridge view controller for Omran Caption (set as the initial controller in
/// Main.storyboard).
///
/// It adds a tiny `keepAwake` message channel so the web app can keep the screen
/// on while live captioning is running. Without it iOS auto-locks the phone after
/// a short idle period, which suspends the WebView and its microphone and stops
/// captioning in the middle of a conversation.
///
/// JS side: `window.webkit.messageHandlers.keepAwake.postMessage(true | false)`
class MainViewController: CAPBridgeViewController {
    private let keepAwakeHandler = KeepAwakeMessageHandler()

    override func capacitorDidLoad() {
        super.capacitorDidLoad()
        webView?.configuration.userContentController.add(keepAwakeHandler, name: KeepAwakeMessageHandler.name)
    }
}

/// Toggles the system idle timer (auto-lock). Kept as a separate object because
/// WKUserContentController retains its handlers strongly.
final class KeepAwakeMessageHandler: NSObject, WKScriptMessageHandler {
    static let name = "keepAwake"

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        let keepOn = (message.body as? Bool) ?? false
        DispatchQueue.main.async {
            UIApplication.shared.isIdleTimerDisabled = keepOn
        }
    }
}
